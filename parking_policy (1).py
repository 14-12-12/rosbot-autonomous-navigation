#!/usr/bin/env python3

import numpy as np
import csv
import os

class ParkingPolicy:
    def __init__(self):
        self.states  = []
        self.actions = []
        self.loaded  = False

        SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        csv_path   = os.path.join(SCRIPT_DIR, "demonstrations.csv")

        if os.path.exists(csv_path):
            self._load(csv_path)
            print("Loaded " + str(len(self.states)) + " samples")
        else:
            print("demonstrations.csv not found!")
            print("Run collect_data.py first!")

    def _load(self, path):
        with open(path, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                state  = np.array([
                    float(row[2]),
                    float(row[3]),
                    float(row[4])
                ])
                action = np.array([
                    float(row[5]),
                    float(row[6])
                ])
                self.states.append(state)
                self.actions.append(action)

        self.states  = np.array(self.states)
        self.actions = np.array(self.actions)
        self.loaded  = True

    def predict(self, x, y, theta):
        if not self.loaded:
            return 0.0, 0.0

        query = np.array([x, y, theta])

        # Compute distance with proper angle wrapping
        diffs = self.states - query
        # Wrap theta difference to [-pi, pi]
        diffs[:, 2] = np.arctan2(
            np.sin(diffs[:, 2]), np.cos(diffs[:, 2]))
        distances = np.linalg.norm(diffs, axis=1)

        # Use K=3 nearest neighbors
        k = min(3, len(self.states))
        nearest_k = np.argsort(distances)[:k]
        action    = np.mean(self.actions[nearest_k], axis=0)

        return float(action[0]), float(action[1])
