import math
import numpy as np
import pandas as pd
import os

from api_client import SphinxAPIClient
from strategy import MortyRescueStrategy
from sinusoid_features import build_features


class LinUCBStrategy(MortyRescueStrategy):

    def __init__(
        self,
        client: SphinxAPIClient,
        d: int = 5,
        alpha_start: float = 0.9,
        alpha_min: float = 0.05,
        alpha_decay: float = 0.008,
        exploration_rounds: int = 30,
        pure_exploitation_threshold: int = 150,
    ):
        super().__init__(client)

        self.feature_dim = d  # dimension des features x
        self.alpha_start = alpha_start
        self.alpha_min = alpha_min
        self.alpha_decay = alpha_decay
        self.exploration_rounds = exploration_rounds
        self.pure_exploitation_threshold = pure_exploitation_threshold

        #  A_p  et b_p pour chaque planète
        self.feature_memory = [np.eye(d) for _ in range(3)]         #  (d x d)
        self.reward_feature_sum = [np.zeros((d, 1)) for _ in range(3)]  # (d x 1)

        self.total_trips = 0
        self.num_trips_per_planet = [0, 0, 0]
        self.run_log = []
        self.alpha_effective = alpha_start
        self.planets = [0, 1, 2]

        self.phase_periods = {0: 10, 1: 20, 2: 200}
        self.phase_success = {p: np.zeros(T) for p, T in self.phase_periods.items()}
        self.phase_trials  = {p: np.zeros(T) for p, T in self.phase_periods.items()}

    def _current_alpha(self) -> float:
        decayed_alpha = self.alpha_start * math.exp(-self.alpha_decay * self.total_trips)
        return max(self.alpha_min, decayed_alpha)

    def _feature_vector(self, planet: int, step: int) -> np.ndarray:
        x = build_features(planet, step, d=self.feature_dim)
        x = np.asarray(x, dtype=float).reshape(-1, 1)  # (d, 1)

        if x.shape[0] != self.feature_dim:
            raise ValueError(f"build_features a renvoyé une dimension {x.shape[0]} mais feature_dim = {self.feature_dim}")
        return x

    def _get_parameter_vector(self, planet: int):

        A_p = self.feature_memory[planet]
        b_p = self.reward_feature_sum[planet]
        A_p_inv = np.linalg.inv(A_p)
        theta = A_p_inv @ b_p
        return theta, A_p_inv

    def _ucb_for_planet(self, planet: int, step: int):
        x = self._feature_vector(planet, step)  # (d, 1)
        theta, A_p_inv = self._get_parameter_vector(planet)
     
        predicted_reward = float(theta.T @ x)
        predicted_prob = max(0.0, min(1.0, predicted_reward))  #  bornée [0,1]
        uncertainty = float(x.T @ A_p_inv @ x)

        alpha = getattr(self, "alpha_effective", self._current_alpha())
        bonus = alpha * math.sqrt(uncertainty)

        ucb_score = predicted_prob + bonus

        T = self.phase_periods[planet]
        phase_idx = step % T
        trials = self.phase_trials[planet][phase_idx]

        if trials >= 5:  
            local_rate = self.phase_success[planet][phase_idx] / trials
            if local_rate < 0.3:
                ucb_score *= 0.2   
            elif local_rate < 0.5:
                ucb_score *= 0.5  
        return ucb_score, predicted_prob
    
    def _decide_morty_count(self, planet: int, predicted_prob: float, step: int) -> int:
        #décider du nombre de Morties à envoyer selon la probabilité prédite de survie et les stats en ligne
        T = self.phase_periods[planet]
        phase_idx = step % T
        trials = self.phase_trials[planet][phase_idx]
        if trials >= 5:
            local_rate = self.phase_success[planet][phase_idx] / trials

            if local_rate < 0.30:
                return 1
            if local_rate < 0.55:
                return 1 if predicted_prob < 0.70 else 2
        if step < 90:
            return 1

        visits = self.num_trips_per_planet[planet]

        if visits < 20:
            return 2 if predicted_prob >= 0.80 else 1

        if predicted_prob >= 0.90:
            return 3
        elif predicted_prob >= 0.70:
            return 2
        else:
            return 1


    def _choose_planet_and_morties(self, step: int):
        # Exploration
        if self.total_trips <= self.exploration_rounds:
            planet = (self.total_trips - 1) % 3
            morty_count = 1
            mode = "explore"
            return planet, morty_count, mode

        # LinUCB
        ucb_scores = []
        predicted_probs = []

        for p in self.planets:
            ucb_p, prob_p = self._ucb_for_planet(p, step)
            ucb_scores.append(ucb_p)
            predicted_probs.append(prob_p)

        best_index = int(np.argmax(ucb_scores))
        planet = self.planets[best_index]
        best_predicted_prob = predicted_probs[best_index]

        morty_count = self._decide_morty_count(planet, best_predicted_prob, step)
        mode = "linUCB"
        return planet, morty_count, mode

    def execute_strategy(self):

        print("\n=== EXECUTING SinusoidFeatureLinUCBStrategy ===")
        status = self.client.get_status()
        morties_remaining = status["morties_in_citadel"]
        print(f"Starting with {morties_remaining} Morties in Citadel")

        while morties_remaining > 0:
            status = self.client.get_status()
            step = status["steps_taken"]
            morties_remaining = status["morties_in_citadel"]

            self.total_trips += 1
            base_alpha = self._current_alpha()
            if morties_remaining < self.pure_exploitation_threshold:
                self.alpha_effective = 0.0
            else:
                self.alpha_effective = base_alpha

            planet, morty_count, mode = self._choose_planet_and_morties(step)
            morties_remaining = status.get("morties_in_citadel", morties_remaining)
            morty_count = min(morty_count, morties_remaining)

            if morty_count <= 0:
                print("No Morties remaining to send ")
                break

            try:
                result = self.client.send_morties(planet, morty_count)
                self.run_log.append({"trip_index": self.total_trips,"mode": mode,"steps_taken": result["steps_taken"],"planet": planet,"planet_name": self.client.get_planet_name(planet),
                    "morty_count": morty_count,"survived": int(result["survived"]),"morties_in_citadel": result["morties_in_citadel"],"morties_on_planet_jessica": result["morties_on_planet_jessica"],
                    "morties_lost": result["morties_lost"]})

                T = self.phase_periods[planet]
                phase_idx = result["steps_taken"] % T
                self.phase_trials[planet][phase_idx] += 1 
                self.phase_success[planet][phase_idx] += int(result["survived"])

            except Exception as e:
                print(f"Error sending morty_count ={morty_count}): {e}")
                try:
                    status = self.client.get_status()
                    morties_remaining = status.get("morties_in_citadel", morties_remaining)
                    print(f"Refreshed remaining morties: {morties_remaining}")

                    if morties_remaining > 0:
                        retry_count = min(morty_count, morties_remaining)
                        print(f"Retrying send with {retry_count} Morty(ies)")
                        result = self.client.send_morties(planet, retry_count)
                    else:
                        break
                except Exception as e2:
                    print(f"Retry failed or cannot refresh status: {e2}")
                    break

            morties_remaining = result["morties_in_citadel"]
            self.num_trips_per_planet[planet] += 1
            reward = 1.0 if result["survived"] else 0.0

            x = self._feature_vector(planet, step)
            self.feature_memory[planet] += x @ x.T
            self.reward_feature_sum[planet] += reward * x

            print(f"Trip {self.total_trips:4d} | mode={mode:8s} | step={result['steps_taken']:3d} | planet={self.client.get_planet_name(planet)} | alpha={self._current_alpha():.3f} | sent={morty_count} | "
                f"survived={int(result['survived'])} | on Jessica={result['morties_on_planet_jessica']:4d} | remaining={morties_remaining:4d} | deaths={result['morties_lost']:4d}")

        final_status = self.client.get_status()
        print("\n=== FINAL RESULTS ===")
        print(f"Morties Saved: {final_status['morties_on_planet_jessica']}")
        print(f"Morties Lost: {final_status['morties_lost']}")
        print(f"Total Steps: {final_status['steps_taken']}")
        print(f"Success Rate: {(final_status['morties_on_planet_jessica']/1000)*100:.2f}%")

        try:
            if self.run_log:
                df_run = pd.DataFrame(self.run_log)
                df_run.to_csv("last_run.csv", index=False)
                print("Run log saved to 'last_run.csv'")
        except Exception as e:
            print(f"Failed to save run log CSV: {e}")

        try:
            for p in self.planets:
                theta, _ = self._get_parameter_vector(p)  
                theta_vec = np.asarray(theta).reshape(-1) 
                np.save(f"theta_planet_{p}.npy", theta_vec)
            print(f"Saved theta files: {[f'theta_planet_{p}.npy' for p in self.planets]}")
        except Exception as e:
            print(f"Failed to save theta files: {e}")
