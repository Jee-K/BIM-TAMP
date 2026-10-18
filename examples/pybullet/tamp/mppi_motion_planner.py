#!/usr/bin/env python3
"""Optional MPPI refinement of IK-generated arm trajectories."""

import numpy as np
import pybullet as p
from examples.pybullet.utils.pybullet_tools.utils import (
    set_joint_positions, pairwise_collision, BodySaver, get_moving_links,
    link_from_name, pairwise_link_collision
)
from examples.pybullet.utils.pybullet_tools.pr2_primitives import (
    create_trajectory, Commands, State
)


class MPPIMotionPlanner:
    """Sample trajectory perturbations and update a nominal joint path."""

    def __init__(self, robot, arm_joints, obstacles,
                 num_samples=100, horizon=20, temperature=0.1,
                 num_iterations=10):
        self.robot = robot
        self.arm_joints = arm_joints
        self.obstacles = obstacles
        self.num_samples = num_samples
        self.horizon = horizon
        self.temperature = temperature
        self.num_iterations = num_iterations

    def plan(self, start_conf, goal_conf, max_time=1.0):
        """
        Plan a collision-free trajectory from start to goal using MPPI.

        Args:
            start_conf: Starting joint configuration
            goal_conf: Goal joint configuration
            max_time: Maximum planning time in seconds

        Returns:
            List of configurations forming the path, or None if no path found
        """
        import time
        start_time = time.time()


        path = self._interpolate(start_conf, goal_conf, self.horizon)

        for iteration in range(self.num_iterations):
            if time.time() - start_time > max_time:
                break


            samples = self._sample_trajectories(path)


            costs = np.array([self._evaluate_trajectory(traj) for traj in samples])


            best_idx = np.argmin(costs)
            if costs[best_idx] < float('inf'):

                return samples[best_idx]


            weights = self._compute_weights(costs)
            path = self._update_path(samples, weights)


        if self._evaluate_trajectory(path) < float('inf'):
            return path

        return None

    def _interpolate(self, start, goal, num_steps):
        """Linear interpolation between start and goal"""
        path = []
        for alpha in np.linspace(0, 1, num_steps):
            conf = [(1-alpha)*s + alpha*g for s, g in zip(start, goal)]
            path.append(conf)
        return path

    def _sample_trajectories(self, nominal_path):
        """Sample trajectories around the nominal path"""
        samples = []
        noise_std = 0.3

        for _ in range(self.num_samples):
            noisy_path = []
            for t, conf in enumerate(nominal_path):

                if t == 0 or t == len(nominal_path) - 1:

                    noisy_path.append(list(conf))
                else:

                    noise = np.random.normal(0, noise_std, len(conf))
                    noisy_conf = [c + n for c, n in zip(conf, noise)]
                    noisy_path.append(noisy_conf)
            samples.append(noisy_path)

        return samples

    def _evaluate_trajectory(self, trajectory):
        """
        Evaluate cost of a trajectory.
        Returns inf if collision, otherwise distance to goal + smoothness cost
        """
        saver = BodySaver(self.robot)


        for conf in trajectory:
            set_joint_positions(self.robot, self.arm_joints, conf)


            arm_links = get_moving_links(self.robot, self.arm_joints)
            base_link = link_from_name(self.robot, 'base_link')
            for link in arm_links:
                if pairwise_link_collision(self.robot, link, self.robot, base_link):
                    saver.restore()
                    return float('inf')


            for obs in self.obstacles:
                if pairwise_collision(self.robot, obs):
                    saver.restore()
                    return float('inf')


        goal_conf = trajectory[-1]
        smoothness = sum(
            sum((trajectory[i+1][j] - trajectory[i][j])**2
                for j in range(len(self.arm_joints)))
            for i in range(len(trajectory)-1)
        )

        saver.restore()
        return smoothness

    def _compute_weights(self, costs):
        """Compute importance weights using exponential transformation"""

        valid_mask = np.isfinite(costs)
        if not np.any(valid_mask):

            return np.ones(len(costs)) / len(costs)


        min_cost = np.min(costs[valid_mask])
        exp_costs = np.zeros(len(costs))
        exp_costs[valid_mask] = np.exp(-(costs[valid_mask] - min_cost) / self.temperature)


        weights = exp_costs / (np.sum(exp_costs) + 1e-10)
        return weights

    def _update_path(self, samples, weights):
        """Update path as weighted average of samples"""
        num_waypoints = len(samples[0])
        num_joints = len(self.arm_joints)

        new_path = []
        for t in range(num_waypoints):

            new_conf = [0.0] * num_joints
            for sample, weight in zip(samples, weights):
                for j in range(num_joints):
                    new_conf[j] += weight * sample[t][j]
            new_path.append(new_conf)


        new_path[0] = samples[0][0]
        new_path[-1] = samples[0][-1]

        return new_path


def mppi_ik_gen(problem, base_ik_gen_fn, use_mppi=True,
                num_samples=50, horizon=10, num_iterations=5):
    """
    MPPI-enhanced IK generator that wraps a base IK generator.

    This is a wrapper that:
    1. Calls base_ik_gen_fn to get initial IK solutions
    2. Uses MPPI to refine/optimize the arm trajectories
    3. Returns improved solutions

    Args:
        problem: The TAMP problem instance
        base_ik_gen_fn: Base IK generator function (e.g., minimal_ik_gen)
        use_mppi: If True, use MPPI to optimize trajectories; if False, just pass through
        num_samples: Number of trajectory samples for MPPI
        horizon: Number of waypoints in MPPI trajectory
        num_iterations: Number of MPPI optimization iterations

    Returns:
        Generator function compatible with PDDLStream
    """
    from examples.pybullet.utils.pybullet_tools.pr2_utils import get_arm_joints


    base_gen = base_ik_gen_fn(problem)

    def gen(arm, obj, pose, grasp):
        """Generator that yields (base_conf, arm_trajectory) tuples"""
        robot = problem.robot
        arm_joints = get_arm_joints(robot, arm)


        obstacles = list(problem.fixed) + [o for o in problem.movable if o != obj]


        for result in base_gen(arm, obj, pose, grasp):
            if result is None:


                yield None
                continue
            bq, cmd = result
            if not use_mppi:

                print(f"[MPPI WRAPPER] Passing through base IK solution for obj {obj}")
                yield (bq, cmd)
                continue


            if not cmd.commands or len(cmd.commands) == 0:

                yield (bq, cmd)
                continue

            traj_cmd = cmd.commands[0]
            if not hasattr(traj_cmd, 'path') or len(traj_cmd.path) < 2:

                yield (bq, cmd)
                continue


            original_path = traj_cmd.path

            if hasattr(original_path[0], 'values'):
                start_conf = list(original_path[0].values)
                goal_conf = list(original_path[-1].values)
            else:

                start_conf = list(original_path[0])
                goal_conf = list(original_path[-1])

            print(f"[MPPI WRAPPER] Optimizing trajectory for obj {obj}: "
                  f"{len(original_path)} waypoints -> MPPI with {horizon} waypoints")


            mppi_planner = MPPIMotionPlanner(
                robot=robot,
                arm_joints=arm_joints,
                obstacles=obstacles,
                num_samples=num_samples,
                horizon=horizon,
                temperature=0.1,
                num_iterations=num_iterations
            )


            saver = BodySaver(robot)
            bq.assign()


            optimized_path = mppi_planner.plan(start_conf, goal_conf, max_time=0.5)

            saver.restore()
            del mppi_planner

            if optimized_path is not None:

                optimized_traj = create_trajectory(robot, arm_joints, optimized_path)
                optimized_cmd = Commands(State(), savers=[BodySaver(robot)],
                                        commands=[optimized_traj])
                optimized_cmd.target = obj

                print(f"[MPPI WRAPPER] Successfully optimized trajectory "
                      f"({len(original_path)} -> {len(optimized_path)} waypoints)")
                yield (bq, optimized_cmd)
            else:

                print(f"[MPPI WRAPPER] MPPI optimization failed, using original trajectory")
                yield (bq, cmd)

    return gen


