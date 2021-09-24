import numpy as np
from scipy.spatial.distance import directed_hausdorff


from MultiVehicleEnv.basic import World, Vehicle,Obstacle,Landmark
from MultiVehicleEnv.scenario import BaseScenario
from MultiVehicleEnv.utils import coord_data_dist, naive_inference, hsv2rgb, coord_dist

'''
use Hausdorff distance as reward function
refer to https://www.wikiwand.com/en/Hausdorff_distance#/Applications
'''

def default_vehicle():
    agent = Vehicle()
    agent.r_safe     = 0.17
    agent.L_car      = 0.25
    agent.W_car      = 0.18
    agent.L_axis     = 0.2
    agent.K_vel      = 0.707
    agent.K_phi      = 0.596
    agent.dv_dt      = 10.0
    agent.dphi_dt    = 10.0
    agent.color      = [[1.0,0.0,0.0],1.0]
    return agent


class Scenario(BaseScenario):
    def make_world(self, args):
        self.args = args
        world = World()

        #for simulate real world
        world.step_t = args.step_t
        world.sim_step = args.sim_step
        world.field_range = [-2.0,-2.0,2.0,2.0]

        num_agent_ray = 60


        # set world.GUI_port as port dir when usegui is true
        if args.usegui:
            world.GUI_port = args.gui_port
        else:
            world.GUI_port = None

        # add 4 agents
        agent_number = args.num_agents

        world.vehicle_list = []
        for idx in range(agent_number):
            vehicle_id = 'AKM_'+str(idx+1)
            agent = default_vehicle()
            agent.vehicle_id = vehicle_id
            agent.ray        = np.zeros((num_agent_ray, 2))
            agent.min_ray    = np.array([0, len(agent.ray)//2])
            agent.color = [hsv2rgb((idx/agent_number)*360,1.0,1.0), 0.8]
            world.vehicle_list.append(agent)
            world.vehicle_id_list.append(vehicle_id)
            world.data_interface[vehicle_id] = {}
            world.data_interface[vehicle_id]['x'] = 0.0
            world.data_interface[vehicle_id]['y'] = 0.0
            world.data_interface[vehicle_id]['theta'] = 0.0

        # add landmark_list
        landmark_number = args.num_landmarks
        world.landmark_list = []
        for idx in range(landmark_number):
            entity = Landmark()
            entity.radius = 0.4
            entity.collideable = False
            entity.color  = [[0.0,1.0,0.0],0.1]
            world.landmark_list.append(entity)

        # add obstacle_list
        obstacle_number = args.num_obstacles
        world.obstacle_list = []
        for idx in range(obstacle_number):
            entity = Obstacle()
            entity.radius = 0.14
            entity.collideable = True
            entity.color  = [[0.0,0.0,0.0],1.0]
            world.obstacle_list.append(entity)

        world.data_slot['max_step_number'] = 20
        # make initial conditions
        self.reset_world(world)
        return world
    
    def observation(self, agent:Vehicle, world:World):
        # agent pos & communication
        entity_pos = []
        for entity in world.landmark_list:
            entity_pos.append(entity.state.coordinate[0])
            entity_pos.append(entity.state.coordinate[1])
        other_pos = []
        comm = []
        for other in world.vehicle_list:
            if other is agent: continue
            comm = comm + other.state.c
            other_pos.append(other.state.coordinate[0] - agent.state.coordinate[0])
            other_pos.append(other.state.coordinate[1] - agent.state.coordinate[1])
        # TODO: remove absolute coordinate
        return np.array([agent.state.vel_b]+agent.state.coordinate+entity_pos + other_pos + comm)

    def reward(self, agent, world, old_world:World = None):
        rew = 0
        u = [a.state.coordinate for a in world.vehicle_list]
        u = u - np.mean(u)
        v = [l.state.coordinate for l in world.landmark_list]
        v = v - np.mean(v)
        rew = -max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        if agent.state.crashed:
            rew -= 1.0
        return rew

    def reset_world(self, world):
        world.data_slot['total_step_number'] = 0
        # set random initial states
        for agent in world.vehicle_list:
            agent.state.theta = np.random.uniform(0,2*3.14159)
            #agent.state.theta = 1/4 * 3.14159 
            agent.state.vel_b = 0
            agent.state.phi = 0
            agent.state.ctrl_vel_b = 0
            agent.state.ctrl_phi = 0
            agent.state.movable = True
            agent.state.crashed = False
        
        # place all landmark,obstacle and vehicles in the field with out conflict
        conflict = True
        while conflict:
            conflict = False
            all_circle = []
            for landmark in world.landmark_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    landmark.state.coordinate[idx] = norm_pos * scale + trans
                all_circle.append((landmark.state.coordinate[0],landmark.state.coordinate[1],landmark.radius))
    
            for obstacle in world.obstacle_list:
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,+1)
                    obstacle.state.coordinate[idx] = norm_pos * scale*0.5 + trans
                all_circle.append((obstacle.state.coordinate[0],obstacle.state.coordinate[1],obstacle.radius))


            for i, agent in enumerate(world.vehicle_list):
                for idx in range(2):
                    scale = world.field_half_size[idx]
                    trans = world.field_center[idx]
                    norm_pos = np.random.uniform(-1,1)
                    #norm_pos = world.ideal_topo_point[idx][i]
                    agent.state.coordinate[idx] = norm_pos * scale + trans
                    #agent.state.coordinate[idx] = norm_pos
                all_circle.append((agent.state.coordinate[0],agent.state.coordinate[1],agent.r_safe))
            
            for idx_a in range(len(all_circle)):
                for idx_b in range(idx_a+1,len(all_circle)):
                    x_a = all_circle[idx_a][0]
                    y_a = all_circle[idx_a][1]
                    r_a = all_circle[idx_a][2]
                    x_b = all_circle[idx_b][0]
                    y_b = all_circle[idx_b][1]
                    r_b = all_circle[idx_b][2]
                    dis = ((x_a - x_b)**2 + (y_a - y_b)**2)**0.5
                    if dis < r_a + r_b:
                        conflict = True
                        break
                if conflict:
                    break
        for agent in world.vehicle_list:
            agent_data = world.data_interface[agent.vehicle_id]
            target_x = world.landmark_list[0].state.coordinate[0]
            target_y = world.landmark_list[0].state.coordinate[1]
            target_data = {'x':target_x, 'y':target_y}
            agent.dis2goal = coord_data_dist(agent_data, target_data)
            agent.dis2goal_prev = agent.dis2goal
        
        for landmark in world.landmark_list:
            landmark.color[1] = 0.1
        # set real landmark and make it color solid
        world.data_slot['real_landmark'] = np.random.randint(len(world.landmark_list))
        real_landmark = world.landmark_list[world.data_slot['real_landmark']]
        real_landmark.color[1] = 1.0
        
        # encode 4 directions into [0,1,2,3]
        def encode_direction(direction):
            if direction[0] > 0 and direction[1] > 0:
                return 0
            if direction[0] < 0 and direction[1] > 0:
                return 1
            if direction[0] < 0 and direction[1] < 0:
                return 2
            if direction[0] > 0 and direction[1] < 0:
                return 3
        # decode 4 direction code [0,1,2,3] into onehot vector
        world.data_slot['direction_decoder'] = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
        # give each agent a direction code as a human hint
        for agent in world.vehicle_list:
            direction = [real_landmark.state.coordinate[0] - agent.state.coordinate[0],
                         real_landmark.state.coordinate[1] - agent.state.coordinate[1]]
            agent.data_slot['direction_obs'] = world.data_slot['direction_decoder'][encode_direction(direction)]

    def info(self, agent:Vehicle, world:World):
        agent_info:dict = {}
        return agent_info

