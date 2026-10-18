from examples.pybullet.utils.pybullet_tools.pr2_primitives import iterate_approach_path
from examples.pybullet.utils.pybullet_tools.utils import pairwise_collision, get_distance, multiply, set_pose, \
    interpolate_poses, invert, wait_if_gui, get_body_name


def distance_fn(q1, q2):


    return get_distance(q1.values[:2], q2.values[:2])


def move_cost_fn(t):
    distance = t.distance(distance_fn=lambda q1, q2: get_distance(q1[:2], q2[:2]))
    return BASE_CONSTANT + distance / BASE_VELOCITY


def get_cfree_pose_pose_test(collisions=True, **kwargs):
    def test(b1, p1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p1.assign()
        p2.assign()
        return not pairwise_collision(b1, b2, **kwargs)
    return test


def get_cfree_obj_approach_pose_test(collisions=True):
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        grasp_pose = multiply(p1.value, invert(g1.value))
        approach_pose = multiply(p1.value, invert(g1.approach), g1.value)
        for obj_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(b1, obj_pose)
            if pairwise_collision(b1, b2):
                return False
        return True
    return test


def get_cfree_approach_pose_test(problem, collisions=True):
    def test(b1, p1, g1, b2, p2):
        if not collisions or (b1 == b2):
            return True
        p2.assign()
        grasp_pose = multiply(p1.value, invert(g1.value))
        approach_pose = multiply(p1.value, invert(g1.approach))
        for obj_pose in interpolate_poses(grasp_pose, approach_pose):
            set_pose(b1, multiply(obj_pose, g1.value))
            if pairwise_collision(b1, b2):
                return False
        return True
    return test


def get_cfree_traj_pose_test(robot, collisions=True):
    def test(c, b2, p2):
        print("[CFREE_TRAJ] called")

        if not collisions:
            return True
        state = c.assign()
        if b2 in state.attachments:
            return True
        p2.assign()
        for _ in c.apply(state):
            state.assign()
            for b1 in state.attachments:
                if pairwise_collision(b1, b2):
                    print(f"[CFREE_TRAJ] collision attached b1={b1} ({get_body_name(b1)}) with "
                          f"b2={b2} ({get_body_name(b2)})")

                    return False
            if pairwise_collision(robot, b2):
                print(f"[CFREE_TRAJ] collision robot={robot} with "
                      f"b2={b2} ({get_body_name(b2)})")
                return False

        return True
    return test


def get_cfree_traj_grasp_pose_test(problem, collisions=True):
    def test(c, a, b1, g, b2, p2):
        raise NotImplementedError()
        if not collisions or (b1 == b2):
            return True
        state = c.assign()
        if (b1 in state.attachments) or (b2 in state.attachments):
            return True
        p2.assign()
        grasp_attachment = g.get_attachment(problem.robot, a)
        for _ in c.apply(state):
            state.assign()
            grasp_attachment.assign()
            if pairwise_collision(b1, b2):
                return False
            if pairwise_collision(problem.robot, b2):
                return False
        return True
    return test
