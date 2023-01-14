import os
import random
import math

import pybullet as p
import numpy as np
import matplotlib.pyplot as plt
from gym import Env
import cv2

ROOM_SIZE_METER = 12

class BoxWorld(Env):
    def __init__(self, max_steps_per_episode=250, GUI=False):

        self.GUI = GUI
        self.client = p.connect(p.GUI) if self.GUI else p.connect(p.DIRECT)

        p.setTimeStep(1. / 240, self.client)
        # p.setGravity(0, 0, -9.8)
        self.load_steps = 100
        if GUI:
            p.resetDebugVisualizerCamera(cameraDistance=8, cameraYaw=0, cameraPitch=-90,
                                         cameraTargetPosition=[ROOM_SIZE_METER / 2, ROOM_SIZE_METER / 2, 8])
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
            p.configureDebugVisualizer(p.COV_ENABLE_WIREFRAME, 1)

        self.agent_name = os.path.join(os.path.dirname(__file__), '../resources/agent.urdf')
        # self.sphere_name = os.path.join(os.path.dirname(__file__), '../resources/sphere2.urdf')
        self.cube_name = "cube"  # os.path.join(os.path.dirname(__file__), '../resources/cube.urdf')
        # self.cylinder_name = os.path.join(os.path.dirname(__file__), '../resources/cylinder.urdf')
        self.cone_name = os.path.join(os.path.dirname(__file__), '../resources/obj_files/cone_blue.obj')
        self.plane_name = os.path.join(os.path.dirname(__file__), '../resources/plane.urdf')

        self.wall_width = 0.1
        self.wall_height = 4

        self.spawn_height = 0.5

        self.num_robots = 1
        self.robot = None

        self.num_objects = 3
        self.objects = []
        self.object_init_pos = []

        self.backpack = [(-1, ""), -1]  # which object is carried on. (pybullet id, shape name), object id

        self.img_w = 80
        self.fov = 90
        self.distance = 20

        self.max_episode_steps = max_steps_per_episode

    def reset(self):
        p.resetSimulation(self.client)
        self.plane = p.loadURDF(fileName=self.plane_name, basePosition=[ROOM_SIZE_METER / 2, ROOM_SIZE_METER / 2, 0])
        self.spawn_wall()

        self.current_step = 0
        self.backpack = [(-1, ""), -1]

        self.robot = None
        self.spawn_robot()

        self.object_init_pos = []
        self.objects = []
        self.spawn_objects(init=True)  # record initial positions

        if self.GUI:
            self.fpv_prev = np.zeros((self.img_w, self.img_w, 3), dtype=np.float32)
            self.fpv_curr = np.zeros((self.img_w, self.img_w, 3), dtype=np.float32)
            self.fpv_depth = np.zeros((self.img_w, self.img_w), dtype=np.float32)
            self.bev = None
            self.load_camera()
            self.bev_target = self.bev

        self.spawn_objects()  # rerandomize locations

        # compute distance from agent to object
        self.prev_distance_agent_to_object = self.compute_distance_agent_to_object()

        state = self.get_state()

        if self.GUI:
            self.load_camera()
            self.save_pictures(init=True)
            self.save_pictures()

        return state

    def get_agent_pos_ori(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        ori = p.getEulerFromQuaternion(ori)[2]
        return pos[0], pos[1], ori

    def compute_distance_agent_to_object(self):
        # the first object is the target object.
        agent_pos, _ = p.getBasePositionAndOrientation(self.robot)
        agent_pos = [agent_pos[0], agent_pos[1], 0]
        obj_pos, _ = p.getBasePositionAndOrientation(self.objects[0][0])
        obj_pos = [obj_pos[0], obj_pos[1], 0]
        dist = ((agent_pos[0] - obj_pos[0]) ** 2 + (agent_pos[1] - obj_pos[1]) ** 2) ** 0.5
        return dist

    def get_state(self):
        state = []
        agent_pos, agent_ori = p.getBasePositionAndOrientation(self.robot)
        # 3 objects * 2 curr_pos = 6. first is the target object to pick up
        for i in range(self.num_objects):
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            dist = ((agent_pos[0]-pos[0])**2+(agent_pos[1]-pos[1])**2) ** 0.5
            state.extend([pos[0], pos[1], dist])
        # 3 agent current position and orientation
        agent_ori = p.getEulerFromQuaternion(agent_ori)[2]
        state.extend([agent_pos[0], agent_pos[1], agent_ori, math.cos(agent_ori), math.sin(agent_ori)])
        state = np.array(state)
        # normalize to [-1, 1]
        state[:11] = state[:11] / (ROOM_SIZE_METER / 2) - 1
        state[11] = state[11] / math.pi

        return state

    def save_pictures(self, init=False):
        """
        Save BEV, FPV(D), SPM, PCD images
        input: state = the current state; target = T/F whether target visualization or visualization after step
        """

        # self.fpv_prev = self.fpv_curr # (25600,)
        # self.fpv_curr = fpv # (25600,)
        # self.fpv_depth = fpv_depth  # (6400,)
        # self.bev = tdv # (102400,)

        if init:
            target_bev = np.reshape(np.clip(self.bev_target, 0, 255), (160, 160, 4)).astype(np.uint8)
            f, axarr = plt.subplots(1, 1)
            axarr.imshow(target_bev)
            axarr.axis('off')
            # axarr[1].imshow(self.goal_SPMs[0])
            # axarr[1].axis('off')
            # axarr[2].imshow(self.goal_SPMs[1])
            # axarr[2].axis('off')
            # axarr[3].imshow(self.goal_SPMs[2])
            # axarr[3].axis('off')
            filename = os.path.join(os.path.dirname(__file__), "../env/imgs/target.png")
            plt.savefig(filename)
            plt.close()
            return

        bev = np.reshape(np.clip(self.bev, 0, 255), (160, 160, 4)).astype(np.uint8)
        fpv_prev = np.reshape(np.clip(self.fpv_prev, 0, 255), (self.img_w, self.img_w, 3)).astype(np.uint8)
        fpv_curr = np.reshape(np.clip(self.fpv_curr, 0, 255), (self.img_w, self.img_w, 3)).astype(np.uint8)
        f, axarr = plt.subplots(2, 2)
        axarr[0][0].imshow(fpv_prev)
        axarr[0][0].axis('off')
        axarr[0][1].imshow(fpv_curr)
        axarr[0][1].axis('off')
        axarr[1][0].imshow(bev)
        axarr[1][0].axis('off')
        axarr[1][1].imshow(np.reshape(self.fpv_depth, (self.img_w, self.img_w)))
        axarr[1][1].axis('off')

        filename = os.path.join(os.path.dirname(__file__), "imgs/timestep" + str(self.current_step) + ".png")
        plt.savefig(filename)
        plt.close()

    def load_camera(self):
        """
        Take one frame of robot fpv and bev
        """
        agent_pos, agent_orn = p.getBasePositionAndOrientation(self.robot)

        yaw = p.getEulerFromQuaternion(agent_orn)[-1]
        xA, yA, zA = agent_pos
        # zA = zA + 0.3 # change vertical positioning of the camera

        xB = xA + math.cos(yaw) * self.distance
        yB = yA + math.sin(yaw) * self.distance
        zB = zA

        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[xA, yA, zA],
            cameraTargetPosition=[xB, yB, zB],
            cameraUpVector=[0, 0, 1.0])

        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=1.5, nearVal=0.02, farVal=self.distance)

        ## WE CAN ENABLE/DISABLE SHADOWS HERE
        robot_fpv = p.getCameraImage(self.img_w, self.img_w,
                                     view_matrix,
                                     projection_matrix,
                                     flags=p.ER_NO_SEGMENTATION_MASK)[2:4]

        bev_view_matrix = p.computeViewMatrix(
            cameraEyePosition=[6, 6, 8],
            cameraTargetPosition=[6, 6, 0.5],
            cameraUpVector=[0, 1, 0])

        bev_projection_matrix = p.computeProjectionMatrixFOV(
            fov=90, aspect=1, nearVal=0.02, farVal=25)

        bev = p.getCameraImage(160, 160,
                               bev_view_matrix, bev_projection_matrix,
                               flags=p.ER_NO_SEGMENTATION_MASK)[2]

        fpv = cv2.cvtColor(np.array(robot_fpv[0], dtype=np.float32),
                           cv2.COLOR_RGBA2RGB).flatten()  # 80x80x4 (RGBA) might want to change this to RGB
        fpv_depth = np.array(robot_fpv[1], dtype=np.float32).flatten()  # 80x80
        bev = np.array(bev, dtype=np.float32).flatten()  # 160x160x4

        self.fpv_prev = self.fpv_curr  # (19200,)
        self.fpv_curr = fpv  # (19200,)
        self.fpv_depth = fpv_depth  # (6400,)
        self.bev = bev  # (76800,)

    def step(self, action):
        reward = 0
        done = False
        info = {}

        # action
        self.fwd_step = np.random.normal(0.15, 0.01)  # todo: check
        self.right_drift = 0  # np.random.normal(move_step/10, move_step/30/10)
        self.turn_step = np.random.normal(math.pi / 18, math.pi / 180)  # 1 degree

        if action == 0:  # move forward
            moved = self.move_agent(self.fwd_step, self.right_drift)
        if action == 1:  # move backward
            moved = self.move_agent(-self.fwd_step, self.right_drift)
        if action == 2:  # turn left
            self.turn_agent(self.turn_step)
        if action == 3:  # turn right
            self.turn_agent(-self.turn_step)
        if action == 4:  # pick
            self.pick()
            done = True

        for i in range(self.load_steps):
            p.stepSimulation()

        # state
        state = self.get_state()
        if self.GUI:
            self.load_camera()
            self.save_pictures()

        # reward
        # compute approaching distance reward
        factor = 1
        curr_distance_agent_to_object = self.compute_distance_agent_to_object()
        reward_approaching = self.prev_distance_agent_to_object - curr_distance_agent_to_object
        reward += factor * reward_approaching
        info.update({"reward_approach": reward_approaching, "distance_to_object": curr_distance_agent_to_object})
        self.prev_distance_agent_to_object = curr_distance_agent_to_object

        # stuck punishment:
        if action in [0, 1] and moved == False:
            # print(moved)
            # reward -= 0.1
            info.update({"move_status": "Stuck and cannot move"})

        # pick reward or punishment
        if done:
            if self.backpack[1] == 0:  # if the object 0 has been picked up
                reward += 10
                info.update({"pick_status": "Successfully picked the target object."})
            else:
                # reward -= 1
                info.update({"pick_status": "Failed to pick the target object."})

        # done
        self.current_step += 1
        if self.current_step >= self.max_episode_steps:
            done = True
            info.update({"step_status": "reached max steps per episode"})

        return state, reward, done, info

    def move_agent(self, fwd_dist, right_drift):
        # todo: add rotation randomness.
        # todo: add try times.
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        target = [pos[0] + math.cos(yaw) * fwd_dist, pos[1] + math.sin(yaw) * fwd_dist, pos[2]]
        # target = [target[0] + math.cos(yaw - math.pi / 2) * right_drift,
        #           target[1] + math.sin(yaw - math.pi / 2) * right_drift, target[2]]

        if self.collision_detection(target) != 1:
            return False

        p.resetBasePositionAndOrientation(self.robot, target, ori)

        return True

    def collision_detection(self, target):
        """
        Checks whether the target coordinate is not colliding with other objects/walls or is outside the map.
        input: target = (x,y)
        output: Returns -1 on wall collision, 1 on success, or the object item in self.objects
        """
        x, y, _ = target
        if x + math.sqrt(0.5) >= ROOM_SIZE_METER or \
                x - math.sqrt(0.5) <= 0 or \
                y + math.sqrt(0.5) >= ROOM_SIZE_METER or \
                y - math.sqrt(0.5) <= 0:
            # print(f"({x},{y}) is outside the map.")
            return -1
        for i in range(self.num_objects):
            # if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            diff = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
            radius = math.sqrt(0.5)
            if self.objects[i][1] == self.cube_name:
                radius = math.sqrt(0.5)
            if diff < math.sqrt(0.5) + radius:
                # print("Something in the way.")
                return self.objects[i], i
        # print("Successful.")
        return 1

    def turn_agent(self, turn_angle):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        new_ori = p.getEulerFromQuaternion(ori)
        new_ori = [new_ori[0], new_ori[1], new_ori[2]]
        new_ori[2] += turn_angle
        new_ori = p.getQuaternionFromEuler(new_ori)

        if self.collision_detection(pos) != 1:
            return False
        p.resetBasePositionAndOrientation(self.robot, pos, new_ori)

    def turn(self, dir):
        """
        Command robot to turn a specific direction
        input: dir = 1 or -1
        """
        theta = 5.0 / 180 * math.pi  # change later to gaussian
        pos_ori = p.getBasePositionAndOrientation(self.robot)
        new_ori = p.getEulerFromQuaternion(pos_ori[1])
        new_ori = [new_ori[0], new_ori[1], new_ori[2]]
        new_ori[2] += theta * dir
        new_ori = p.getQuaternionFromEuler(new_ori)
        p.resetBasePositionAndOrientation(self.robot, pos_ori[0], new_ori)

    def pick(self):
        """
        Robot grabs object in fpv and stores in virtual backpack
        """
        info = {"pick_status": "Pick failed bc nothing near"}
        # if self.backpack[1] != -1:
        #     info = {"pick_status": "Pick failed bc backpack is full"}
        #     return info
        grab_object = self.check_grab()
        if grab_object != -1:
            info = {"pick_status": "Picked properly"}
            obj = grab_object[0]
            idx = grab_object[1]
            pos, ori = p.getBasePositionAndOrientation(obj[0])
            pos = [pos[0], pos[1], -10]
            p.resetBasePositionAndOrientation(obj[0], pos, ori)
            self.backpack[0] = obj
            self.backpack[1] = idx
        return info

    def check_grab(self):
        pos, ori = p.getBasePositionAndOrientation(self.robot)
        yaw = p.getEulerFromQuaternion(ori)[2]
        grab_dist = 0.5**0.5  # 1
        x_delta = math.cos(yaw) * grab_dist
        y_delta = math.sin(yaw) * grab_dist
        grab_pos = [pos[0] + x_delta, pos[1] + y_delta, 0]
        for i in range(self.num_objects):
            if i == self.backpack[1]: continue
            pos, _ = p.getBasePositionAndOrientation(self.objects[i][0])
            dist = math.sqrt((pos[0] - grab_pos[0]) ** 2 + (pos[1] - grab_pos[1]) ** 2)
            grab_radius = 1  # 0.6
            if dist < grab_radius:
                return self.objects[i], i
        return -1

    def spawn_wall(self):
        """
        **FROM SPATIAL ACTION MAPS GITHUB**
        spawns the four surroundings walls
        """
        obstacle_color = (1, 1, 1, 1)
        obstacles = []
        for x, y, length, width in [
            (-self.wall_width, 6, self.wall_width, ROOM_SIZE_METER + self.wall_width),
            (ROOM_SIZE_METER + self.wall_width, 6, self.wall_width, ROOM_SIZE_METER + self.wall_width),
            (6, -self.wall_width, ROOM_SIZE_METER + self.wall_width, self.wall_width),
            (6,ROOM_SIZE_METER + self.wall_width, ROOM_SIZE_METER + self.wall_width, self.wall_width)
        ]:
            obstacles.append({'type': 'wall', 'position': (x, y), 'heading': 0, 'length': length, 'width': width})

        seen = []
        dir_path = os.path.dirname(os.path.realpath(__file__))
        filename = dir_path + "/../resources/wall_texture/wall_checkerboard_"
        for obstacle in obstacles:
            obstacle_half_extents = [obstacle['length'] / 2, obstacle['width'] / 2, self.wall_height]
            obstacle_collision_shape_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=obstacle_half_extents)
            obstacle_visual_shape_id = p.createVisualShape(p.GEOM_BOX, halfExtents=obstacle_half_extents,
                                                           rgbaColor=obstacle_color)
            obstacle_id = p.createMultiBody(
                0, obstacle_collision_shape_id, obstacle_visual_shape_id,
                [obstacle['position'][0], obstacle['position'][1], 0.5],
                p.getQuaternionFromEuler([0, 0, obstacle['heading']])
            )
            while True:
                id = random.randint(0, 199)
                if id not in seen:
                    seen.append(id)
                    break
            x = p.loadTexture(filename + str(id) + ".png")
            p.changeVisualShape(obstacle_id, -1, textureUniqueId=x)

    def randomize_objs_pos(self):
        """
        Chooses random locations for the objects
        output: list size n for n objects with each item being (x,y)
        """
        robot_pos = p.getBasePositionAndOrientation(self.robot)[0]
        lastpos = [(robot_pos[0], robot_pos[1], self.spawn_height)]
        for i in range(self.num_objects):
            x = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
            y = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
            j = 0
            while j < i + self.num_robots:
                pos = lastpos[j]
                diff = math.sqrt((pos[0] - x) ** 2 + (pos[1] - y) ** 2)
                if diff <= 2 * math.sqrt(0.5):
                    x = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
                    y = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
                    j = -1
                j += 1
            pos = [x, y, self.spawn_height]
            lastpos.append(pos)
        return lastpos[self.num_robots:]

    def spawn_robot(self):
        # randomize agent's position
        x = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
        y = random.uniform(math.sqrt(0.5), ROOM_SIZE_METER - math.sqrt(0.5))
        pos = [x, y, self.spawn_height]
        ori = p.getQuaternionFromEuler([0, 0, random.uniform(-math.pi, math.pi)])
        self.robot = p.loadURDF(fileName=self.agent_name, basePosition=pos, baseOrientation=ori)

    def spawn_objects(self, init=False):
        """
        spawns the objects within the walls and no collision
        input: init = False; if True, store the init values
        """
        obj_poses = self.randomize_objs_pos()
        obj_colors = [(1, 0, 0, 1), (0, 1, 0, 1), (0, 0, 1, 1), (0, 1, 1, 1), (1, 0, 1, 1), (1, 1, 0, 1)]
        if init:
            self.object_init_pos = obj_poses
            for i in range(self.num_objects):
                choice = random.randint(0, 3)  # object shape
                pos = obj_poses[i]
                color_choice = random.randint(0, len(obj_colors) - 1)
                color = obj_colors[color_choice]
                if choice == 0:  # sphere
                    sphere_collision_id = p.createCollisionShape(p.GEOM_SPHERE)
                    sphere_visual_id = p.createVisualShape(p.GEOM_SPHERE, radius=0.5, rgbaColor=color)
                    sphere_id = p.createMultiBody(0, sphere_collision_id, sphere_visual_id, pos)
                    self.objects.append((sphere_id, "sphere"))
                elif choice == 1:  # cube
                    yaw = random.uniform(0, 2 * math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                    cube_collision_id = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
                    cube_visual_id = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=color)
                    cube_id = p.createMultiBody(0, cube_collision_id, cube_visual_id, pos, ori)
                    self.objects.append((cube_id, self.cube_name))
                elif choice == 2:  # cylinder
                    cylinder_collision_id = p.createCollisionShape(p.GEOM_CYLINDER)
                    cylinder_visual_id = p.createVisualShape(p.GEOM_CYLINDER, radius=0.5, rgbaColor=color)
                    cylinder_id = p.createMultiBody(0, cylinder_collision_id, cylinder_visual_id, pos)
                    self.objects.append((cylinder_id, "cylinder"))
                elif choice == 3:  # cone, no built-in, use obj file
                    ori = [math.pi / 2, 0, 0]
                    ori = p.getQuaternionFromEuler(ori)
                    cone_collision_id = p.createCollisionShape(p.GEOM_MESH, fileName=self.cone_name,
                                                               meshScale=[0.5, 0.5, 0.5])
                    cone_visual_id = p.createVisualShape(p.GEOM_MESH, fileName=self.cone_name,
                                                         meshScale=[0.5, 0.5, 0.5], rgbaColor=color)
                    cone_id = p.createMultiBody(0, cone_collision_id, cone_visual_id, pos, ori)
                    self.objects.append((cone_id, "cone"))
        else:
            for i in range(self.num_objects):
                _, ori = p.getBasePositionAndOrientation(self.objects[i][0])
                if self.objects[i][1] == self.cube_name:
                    # yaw = random.uniform(0, 2*math.pi)
                    # ori = [0, 0, yaw]
                    # ori = p.getQuaternionFromEuler(ori)
                    yaw = random.uniform(0, 2 * math.pi)
                    ori = [0, 0, yaw]
                    ori = p.getQuaternionFromEuler(ori)
                p.resetBasePositionAndOrientation(self.objects[i][0], obj_poses[i], ori)
