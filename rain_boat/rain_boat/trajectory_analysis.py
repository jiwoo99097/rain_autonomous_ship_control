#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

class TrajectoryAnalysis:
    def __init__(self):
        # 데이터 폴더 경로 설정
        self.data_folder = '/home/rain/dock_ws/'

        # 노이즈 수준 리스트
        self.noise_levels = ['10%', '30%', '50%', '70%', '90%']

        # 궤적 데이터를 저장할 딕셔너리 초기화
        self.trajectories = {}
        for noise_level in self.noise_levels:
            self.trajectories[noise_level] = []

        # 경로 추종 성능 지표를 저장할 딕셔너리 초기화
        self.performance_metrics = {noise_level: {} for noise_level in self.noise_levels}

        # 글로벌 경로 로드
        self.global_path = self.load_global_paths()

        # 궤적 데이터 로드 및 평균 궤적 계산
        self.load_and_process_trajectories()

        # 평균 궤적 시각화
        self.plot_average_trajectories()

    def load_global_paths(self):
        # 여러 개의 글로벌 경로 파일을 로드하고 평균을 계산
        file_list = [f for f in os.listdir(self.data_folder) if f.startswith('global_path')]
        global_paths = []
        for filename in file_list:
            filepath = os.path.join(self.data_folder, filename)
            df = pd.read_csv(filepath)
            x = df['X'].values
            y = df['Y'].values
            global_paths.append((x, y))
        # 글로벌 경로 재샘플링 및 평균 계산
        resampled_global_paths = self.resample_all_trajectories(global_paths)
        avg_x, avg_y = self.compute_average_trajectory(resampled_global_paths)
        global_path = list(zip(avg_x, avg_y))
        print(f"Loaded and averaged {len(file_list)} global paths.")
        return global_path

    def load_trajectories(self, noise_level):
        folder_path = self.data_folder
        if noise_level == 'none':
            file_pattern = 'trajectory_experiment_none_'
        else:
            file_pattern = f"trajectory_experiment_{noise_level}_"

        trajectories = []
        file_list = [f for f in os.listdir(folder_path) if f.startswith(file_pattern) and f.endswith('.csv')]
        file_list.sort()
        for filename in file_list:
            filepath = os.path.join(folder_path, filename)
            df = pd.read_csv(filepath)
            x = df['X'].values
            y = df['Y'].values
            trajectories.append((x, y))
        print(f"Loaded {len(trajectories)} trajectories for noise level {noise_level}")
        return trajectories

    def resample_trajectory(self, x, y, num_points=100):
        distances = np.cumsum(np.sqrt(np.diff(x, prepend=x[0])**2 + np.diff(y, prepend=y[0])**2))
        if distances[-1] == 0:
            # 궤적의 길이가 0인 경우 예외 처리
            return np.full(num_points, x[0]), np.full(num_points, y[0])
        distances = distances / distances[-1]
        fx = interp1d(distances, x, kind='linear', fill_value="extrapolate")
        fy = interp1d(distances, y, kind='linear', fill_value="extrapolate")
        uniform_distances = np.linspace(0, 1, num_points)
        x_resampled = fx(uniform_distances)
        y_resampled = fy(uniform_distances)
        return x_resampled, y_resampled

    def resample_all_trajectories(self, trajectories, num_points=100):
        resampled_trajectories = []
        for x, y in trajectories:
            x_resampled, y_resampled = self.resample_trajectory(x, y, num_points)
            resampled_trajectories.append((x_resampled, y_resampled))
        return resampled_trajectories

    def compute_average_trajectory(self, resampled_trajectories):
        x_array = np.array([traj[0] for traj in resampled_trajectories])
        y_array = np.array([traj[1] for traj in resampled_trajectories])
        x_mean = np.mean(x_array, axis=0)
        y_mean = np.mean(y_array, axis=0)
        return x_mean, y_mean

    def calculate_curvature(self, x, y):
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
        curvature = np.nan_to_num(curvature)
        return curvature
    
    def load_and_process_trajectories(self):
        for noise_level in self.noise_levels:
            trajectories = self.load_trajectories(noise_level)
            if trajectories:
                resampled = self.resample_all_trajectories(trajectories)
                avg_x, avg_y = self.compute_average_trajectory(resampled)
                self.trajectories[noise_level] = (avg_x, avg_y)
            else:
                print(f"No trajectories found for noise level {noise_level}")

    def plot_average_trajectories(self):
        # 전역 경로의 x, y 좌표 추출
        global_path_x = [wp[0] for wp in self.global_path]
        global_path_y = [wp[1] for wp in self.global_path]

        plt.figure(figsize=(10, 8))

        # 글로벌 경로 그리기
        plt.plot(global_path_x, global_path_y, 'k-', label='Global Path')

        # 글로벌 경로에서 시작 지점, 중간 지점, 마지막 지점 계산 및 표시
        # 시작 지점
        start_x = global_path_x[0]
        start_y = global_path_y[0]
        plt.plot(start_x, start_y, marker='o', color='k', markersize=8)
        plt.text(start_x, start_y, 'Start', fontsize=9, color='k')

        # 마지막 지점
        end_x = global_path_x[-1]
        end_y = global_path_y[-1]
        plt.plot(end_x, end_y, marker='s', color='k', markersize=8)
        plt.text(end_x, end_y, 'End', fontsize=9, color='k')

        # 곡률 계산하여 중간 지점 찾기
        curvature = self.calculate_curvature(np.array(global_path_x), np.array(global_path_y))
        max_curvature_idx = np.argmax(curvature)
        mid_x = global_path_x[max_curvature_idx]
        mid_y = global_path_y[max_curvature_idx]

        # 중간 지점 표시
        plt.plot(mid_x, mid_y, marker='^', color='k', markersize=8)
        plt.text(mid_x, mid_y, 'Mid', fontsize=9, color='k')

        # 각 노이즈 수준의 평균 궤적 그리기
        colors = ['b', 'g', 'r', 'c', 'm', 'y']
        for idx, noise_level in enumerate(self.noise_levels):
            avg_trajectory = self.trajectories[noise_level]
            if avg_trajectory:
                x_mean, y_mean = avg_trajectory
                label = f'Noise Level {noise_level}'
                plt.plot(x_mean, y_mean, linestyle='--', color=colors[idx % len(colors)], label=label)

        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Comparison of Global Path and Average Trajectories at Different Noise Levels')
        plt.legend()
        plt.grid(True)
        plt.show()
        
if __name__ == '__main__':
    analysis = TrajectoryAnalysis()
