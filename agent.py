from __future__ import division
from shapely.geometry import Point, LineString, Polygon, LinearRing, MultiPolygon
from shapely import strtree, affinity
import math
import numpy as np
import random
from dataclasses import dataclass
from typing import Sequence, Mapping, Dict
from scipy import interpolate

from commonroad.scenario.lanelet import LaneletNetwork
from dg_commons import PlayerName
from dg_commons.sim.goals import PlanningGoal
from dg_commons.sim import SimObservations, InitSimObservations
from dg_commons.sim.agents import Agent
from dg_commons.sim.models.obstacles import StaticObstacle
from dg_commons.sim.models.vehicle import VehicleCommands
from dg_commons.sim.models.vehicle_structures import VehicleGeometry
from dg_commons.sim.models.vehicle_utils import VehicleParameters

from dg_commons.sim.agents.lane_follower import LFAgent
from dg_commons.maps.lanes import DgLanelet, LaneCtrPoint
from dg_commons import SE2Transform, valmap, relative_pose
from dg_commons.controllers.pure_pursuit import PurePursuit, PurePursuitParam
from dg_commons.controllers.speed import SpeedBehavior, SpeedController
from dg_commons.sim.models import kmh2ms, extract_pose_from_state, extract_vel_from_state
from dg_commons.sim.simulator_structures import PlayerObservations
from geometry import SE2value

from shapely.errors import ShapelyDeprecationWarning
import warnings
warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)


@dataclass(frozen=True)
class Pdm4arAgentParams:
    param1: float = 0.2


class Pdm4arAgent(Agent):
    """This is the PDM4AR agent.
    Do *NOT* modify the naming of the existing methods and the input/output types.
    Feel free to add additional methods, objects and functions that help you to solve the task"""

    def __init__(self,
                 sg: VehicleGeometry,
                 sp: VehicleParameters
                 ):
        self.sg = sg  # geometry of car
        self.sp = sp  # max velocity...etc
        self.name: PlayerName = None
        self.goal: PlanningGoal = None
        self.lanelet_network: LaneletNetwork = None
        self.static_obstacles: Sequence[StaticObstacle] = None
        self.new_obstacles: list = None
        self.init_path = True  # indicate if an initial path needs to be computed
        self.path: list = None
        self.remaining_path: list = None
        # feel free to remove/modify  the following
        self.params = Pdm4arAgentParams()
        self.init_obs: InitSimObservations = None

        #####################
        self.rrt: RRTFamilyPathPlanner = None
        self.agent_controller: LFAgent = None
        self.emergency: bool = None
        self.start_timer: bool = False
        self.timer: float = 0.0
        self.other_car: Polygon = None
        self.bypassing: bool = False
        self.yield_face_to_face: bool = True
        self.timer_bypassing: float = 0.0
        self.beginning: bool = True
        self.timer_beginning: float = 0.0

    def on_episode_init(self, init_obs: InitSimObservations):
        """This method is called by the simulator at the beginning of each episode."""
        self.name = init_obs.my_name
        self.goal = init_obs.goal
        self.lanelet_network = init_obs.dg_scenario.lanelet_network
        self.static_obstacles = list(
            init_obs.dg_scenario.static_obstacles.values())
        self.circle_road()
        self.init_obs = init_obs
        # print('static obstacles')

    def circle_road(self):  #ensure that the car cannot bypass the road
        new_obstacles = []
        circle = []
        for obst in self.static_obstacles:
            if isinstance(obst.shape, Polygon):
                new_obstacles.append(Polygon([(obst.shape.exterior.coords.xy[0][i], obst.shape.exterior.coords.xy[1][i]) for i in range(len(obst.shape.exterior.coords.xy[0]))]))
            elif isinstance(obst.shape, LinearRing):
                new_obstacles.append(LinearRing([(obst.shape.coords.xy[0][i], obst.shape.coords.xy[1][i]) for i in range(len(obst.shape.coords.xy[0]))]))
            else:
                circle.extend([(obst.shape.coords.xy[0][i], obst.shape.coords.xy[1][i]) for i in range(len(obst.shape.coords.xy[0]))])
                new_obstacles.append(LineString([(obst.shape.coords.xy[0][i], obst.shape.coords.xy[1][i]) for i in range(len(obst.shape.coords.xy[0]))]))
        # print(circle)
        exterior = [(-50,-50),(-50,80),(45,80),(45,-50)]
        new_obstacles.append(Polygon(exterior,holes=[circle]))
        self.new_obstacles = new_obstacles

    def get_commands(self, sim_obs: SimObservations) -> VehicleCommands:
        """ This method is called by the simulator at each time step.
        For instance, this is how you can get your current state from the observations:
        my_current_state: VehicleState = sim_obs.players[self.name].state

        :param sim_obs:
        :return:
        """
        # todo implement here some better planning
        # print(sim_obs.players[self.name].state.psi)
        if self.init_path:
            init_pos = (sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y)
            self.path = self.get_initial_path(init_pos)
            
            print(self.name)
            print(self.path)
            self.init_path = False

            self.init_controller()

            return VehicleCommands(acc=0, ddelta=0)


        state = sim_obs.players[self.name].state
        self.agent_controller.speed_behavior.update_observations(sim_obs.players)
        self.update_remaining_path(state)

        if not self.path:   #if path is not found
            return VehicleCommands(acc=1.0, ddelta=0)   #if the path is not found just move forward to avoid [ValueError]

        if self.beginning:
            self.timer_beginning += 0.1
        if self.timer_beginning > 5:
            self.beginning = False
        
        if self.timer_bypassing > 8:    #bypassing is over
            self.yield_face_to_face = True
            self.timer_bypassing = 0
            self.bypasing = False
            self.other_car = None
        
        if self.timer >= 4 and self.other_car is not None and not self.yield_face_to_face: #wait 4 seconds in case of emergency and verify that there is another car
            # print('self.other_car', [(self.other_car.exterior.coords.xy[0][i], self.other_car.exterior.coords.xy[1][i]) for i in range(len(self.other_car.exterior.coords.xy[0]))])
            # print('my car occupancy', [(sim_obs.players[self.name].occupancy.exterior.coords.xy[0][i], sim_obs.players[self.name].occupancy.exterior.coords.xy[1][i]) for i in range(len(sim_obs.players[self.name].occupancy.exterior.coords.xy[0]))])
            other_cars = []
            for other_name, _ in sim_obs.players.items():
                if other_name == self.name:
                    continue
                other_cars.append(sim_obs.players[other_name].occupancy)
            init_pose = (sim_obs.players[self.name].state.x, sim_obs.players[self.name].state.y)
            path = self.rrt.replanning(init_pose,self.remaining_path, state.psi, self.other_car, other_cars)
            if path is not None:
                self.path = path
                self.remaining_path = path
                self.init_controller()
                # print(new_bounds)
                print(self.path)
                print('new path computed')
                self.bypassing = True
                self.start_timer = False
                self.timer = 0
                self.agent_controller.speed_behavior.params.nominal_speed = kmh2ms(10)
                
        #Timer if 2 cars are face to face
        if self.start_timer and not self.yield_face_to_face:
            self.timer += 0.1
            if self.timer<5: print(self.name, self.timer, sim_obs.players[self.name].state.vx, self.sp.acc_limits[0])
            # if sim_obs.players[self.name].state.vx > 0: #ensure to break as much as possible
            #     return VehicleCommands(acc=self.sp.acc_limits[0], ddelta=0)
            
        #Check if after a car was in emergency and the other is away
        dist_other_agent = float('inf')
        for other_name, _ in sim_obs.players.items():
            if other_name == self.name:
                continue
            other_state = sim_obs.players[other_name].state
            dist = np.sqrt((other_state.x-state.x)**2 + (other_state.y-state.y)**2)
            if dist < dist_other_agent:
                dist_other_agent = dist
        if self.agent_controller._emergency and dist_other_agent > 10:   #if nobody is around the car
            self.agent_controller._emergency = False
            self.timer = 0
            self.start_timer = False
            self.other_car = None

        # #Check is someone is in front the car 
        someone_in_front, occupancy = self.someone_in_front(sim_obs.players)
        if someone_in_front:
            if sim_obs.players[self.name].occupancy.centroid.xy[1][0] > occupancy.centroid.xy[1][0]:
                self.start_timer = True
                self.yield_face_to_face = False
            else:
                self.start_timer = False
                self.yield_face_to_face = True
            self.agent_controller._emergency = True
            self.other_car = occupancy
            self.timer_bypassing = 0
            self.timer = 0

        
        #JUST FOLLOW THE PATH
        #checker wether the car is in a turn or not in order to adapt pace
        if self.bypassing:
            self.agent_controller._emergency = False
            self.timer_bypassing += 0.1
            self.agent_controller.speed_behavior.params.nominal_speed = kmh2ms(10)
        elif self.beginning:
            self.agent_controller.speed_behavior.params.nominal_speed = kmh2ms(10)
        else:
            dist_point = min([np.sqrt((state.x-p[0])**2+(state.y-p[1])**2) for p in self.path])
            if dist_point < 5: #close to a turn
                self.agent_controller.speed_behavior.params.nominal_speed = kmh2ms(20)
            else:
                self.agent_controller.speed_behavior.params.nominal_speed = kmh2ms(25)
        
        command = self.agent_controller.get_commands(sim_obs)
        # print('self.agent_controller._emergency after get_command ', self.name, self.agent_controller._emergency)
        
        return VehicleCommands(acc=command.acc, ddelta=command.ddelta)
        # return VehicleCommands(acc=self.controller.throttle, ddelta=self.controller.steer_rate)

    def get_initial_path(self, init_pos: tuple) -> list:
        self.rrt = RRTFamilyPathPlanner()
        bounds = self.get_boundary_map()
        goal_region = self.goal.goal
        object_radius = 2
        steer_distance = float('inf')
        num_iter = 20000  # max iter 
        min_iter = 2000 #ensure to have a good sol
        resolution = 16
        print('init_pos', init_pos)#, '\ngoal_region', goal_region)
        path, v, e = self.rrt.path(self.new_obstacles, bounds, init_pos, goal_region,
                              object_radius, steer_distance, num_iter, min_iter, resolution, False, 'RRT*')
        # print(e)
        self.remaining_path = path
        return path

    def init_controller(self):
        pure_pursuit_param = PurePursuitParam(look_ahead_minmax=(1,2), length=self.sg.lf+self.sg.lr, lr=self.sg.lr)
        pure_pursuit = PurePursuit(pure_pursuit_param)
        speed_behavior = SpeedBehavior(self.name)
        speed_behavior.params.nominal_speed = kmh2ms(10)
        speed_behavior.params.yield_distance = 10
        speed_behavior.params.minimum_yield_vel = 1
        speed_behavior.params.safety_time_braking = 2
        lane = []
        for i in range(len(self.path)):
            if i == len(self.path)-1:
                theta = np.arctan2(self.path[i][1]-self.path[i-1][1], self.path[i][0]-self.path[i-1][0])
            else:
                theta = np.arctan2(self.path[i+1][1]-self.path[i][1], self.path[i+1][0]-self.path[i][0])
            lane.append(LaneCtrPoint(SE2Transform(
                [self.path[i][0], self.path[i][1]], theta), self.sg.w_half))
        self.agent_controller = LFAgent(DgLanelet(lane), self.sp, self.sg, speed_behavior=speed_behavior, pure_pursuit=pure_pursuit)
        self.agent_controller.on_episode_init(self.init_obs)

    def get_boundary_map(self) -> tuple:  # min x, min y, max x, and max y
        xmin = float('inf')
        ymin = float('inf')
        xmax = -float('inf')
        ymax = -float('inf')
        for obst in self.static_obstacles:
            if isinstance(obst.shape, Polygon):
                xmin_obst = min(obst.shape.exterior.coords.xy[0])
                ymin_obst = min(obst.shape.exterior.coords.xy[1])
                xmax_obst = max(obst.shape.exterior.coords.xy[0])
                ymax_obst = max(obst.shape.exterior.coords.xy[1])
            else:
                xmin_obst = min(obst.shape.coords.xy[0])
                ymin_obst = min(obst.shape.coords.xy[1])
                xmax_obst = max(obst.shape.coords.xy[0])
                ymax_obst = max(obst.shape.coords.xy[1])
            if xmin_obst < xmin:
                xmin = xmin_obst
            if ymin_obst < ymin:
                ymin = ymin_obst
            if xmax_obst > xmax:
                xmax = xmax_obst
            if ymax_obst > ymax:
                ymax = ymax_obst
        return (xmin-10, ymin-10, xmax+10, ymax+10)

    def someone_in_front(self, agents: Mapping[PlayerName, PlayerObservations]) -> bool:
        mypose = extract_pose_from_state(agents[self.name].state)

        def rel_pose(other_obs: PlayerObservations) -> SE2Transform:
            other_pose: SE2value = extract_pose_from_state(other_obs.state)
            return SE2Transform.from_SE2(relative_pose(mypose, other_pose))

        agents_rel_pose: Dict[PlayerName, SE2Transform] = valmap(rel_pose, agents)
        myvel = agents[self.name].state.vx
        for other_name, _ in agents.items():
            if other_name == self.name:
                continue
            if self.other_car is not None:
                if self.other_car.centroid.xy[1][0] == agents[other_name].occupancy.centroid.xy[1][0] and self.other_car.centroid.xy[0][0] == agents[other_name].occupancy.centroid.xy[0][0]:
                    continue    #the car we are looking is the one faced to us
            rel = agents_rel_pose[other_name]
            other_vel = extract_vel_from_state(agents[other_name].state)
            rel_distance = np.linalg.norm(rel.p)
            in_front_of_me: bool = rel.p[0] > 0 and -2.5 <= rel.p[1] <= 2.5
            # coming_from_the_front: bool = 3 * math.pi / 4 <= abs(rel.theta) <= math.pi * 5 / 4 and in_front_of_me
            # print(self.name, 'in_front_of_me', in_front_of_me, 'coming_from_the_front', coming_from_the_front)
            # if (coming_from_the_front and rel_distance < 2 * (myvel + other_vel)):
            if (in_front_of_me and rel_distance < 2 * (myvel + other_vel)):
                return True, agents[other_name].occupancy
        return False, None
    
    def get_bounds_btw_2cars(self, other_car: Polygon, my_car: Polygon) -> tuple: # min x, min y, max x, and max y
        xmin = min(min(other_car.exterior.coords.xy[0]), min(my_car.exterior.coords.xy[0]))
        ymin = min(min(other_car.exterior.coords.xy[1]), min(my_car.exterior.coords.xy[1]))
        xmax = max(max(other_car.exterior.coords.xy[0]), max(my_car.exterior.coords.xy[0]))
        ymax = max(max(other_car.exterior.coords.xy[1]), max(my_car.exterior.coords.xy[1]))

        return (xmin, ymin, xmax, ymax)

    def update_remaining_path(self, state):
        for pt in self.path:
            dist = np.sqrt((state.x-pt[0])**2 + (state.y-pt[1])**2)
            if dist<20:
                new_remaining_path = [rem_pt for rem_pt in self.remaining_path if rem_pt!=pt]
                self.remaining_path = new_remaining_path
"""
*    Title: Sampling-Based-Path-Planning-Library
*    Author: yrouben
*    Date: 28.12.2022
*    Availability: https://github.com/yrouben/Sampling-Based-Path-Planning-Library/blob/461b3f62e03ded2a99898a35e24a01f4ff68711a/RRTFamilyOfPlanners.py
*    Note that the code is slightly modified in order to be compatible with the project
*
"""

class RRTFamilyPathPlanner():

    """Plans path using an algorithm from the RRT family.
    Contains methods for simple RRT based search, RRTstar based search and informed RRTstar based search.
    """

    def initialise(self, obstacles, bounds, start_pose, goal_region, object_radius, steer_distance, num_iterations, min_iter, resolution, runForFullIterations):
        """Initialises the planner with information about the environment and parameters for the rrt path planers
        Args:
            Obstacles: list of obstacles.
            bounds( (int int int int) ): min x, min y, max x, and max y coordinates of the bounds of the world.
            start_pose( (float float) ): Starting x and y coordinates of the object in question.
            goal_region (Polygon): A polygon representing the region that we want our object to go to.
            object_radius (float): Radius of the object.
            steer_distance (float): Limits the length of the branches
            num_iterations (int): How many points are sampled for the creationg of the tree
            resolution (int): Number of segments used to approximate a quarter circle around a point.
            runForFullIterations (bool): If True RRT and RRTStar return the first path found without having to sample all num_iterations points.
        Returns:
            None
        """
        self.obstacles = obstacles
        self.bounds = bounds
        self.minx, self.miny, self.maxx, self.maxy = bounds
        self.start_pose = start_pose
        goal = goal_region.difference(obstacles[-1])  #last element is the part out of the road
        if isinstance(goal, MultiPolygon):  #if intersection return several polygons
            polygons = list(goal)
            self.goal_region = polygons[np.argmax([pol.area for pol in polygons])]  #goal region = pol with the biggest area
        else:
            self.goal_region = goal
        print([(self.goal_region.exterior.coords.xy[0][i], self.goal_region.exterior.coords.xy[1][i]) for i in range(len(self.goal_region.exterior.coords.xy[0]))])
        # self.tree = strtree.STRtree(self.obstacles)
        self.obj_radius = object_radius
        self.N = num_iterations
        self.min_iter = min_iter
        self.resolution = resolution
        self.steer_distance = steer_distance
        self.V = set()
        self.E = set()
        self.child_to_parent_dict = dict()  # key = child, value = parent
        self.runForFullIterations = runForFullIterations
        self.goal_pose = (goal_region.centroid.coords[0])
        self.path_found = False #used to ensure that the random point at centroid is not called twice & to stop when min_iter is reached
        self.random = False
        # self.replan = False


    def path(self, obstacles, bounds, start_pose, goal_region, object_radius, steer_distance, num_iterations, min_iter, resolution, runForFullIterations, RRT_Flavour):
        """Returns a path from the start_pose to the goal region in the current environment using the specified RRT-variant algorithm.
        Args:
            Obstacles: list of obstacles.
            bounds( (int int int int) ): min x, min y, max x, and max y coordinates of the bounds of the world.
            start_pose( (float float) ): Starting x and y coordinates of the object in question.
            goal_region (Polygon): A polygon representing the region that we want our object to go to.
            object_radius (float): Radius of the object.
            steer_distance (float): Limits the length of the branches
            num_iterations (int): How many points are sampled for the creationg of the tree
            resolution (int): Number of segments used to approximate a quarter circle around a point.
            runForFullIterations (bool): If True RRT and RRTStar return the first path found without having to sample all num_iterations points.
            RRT_Flavour (str): A string representing what type of algorithm to use.
                               Options are 'RRT', 'RRT*', and 'InformedRRT*'. Anything else returns None,None,None.
        Returns:
            path (list<(int,int)>): A list of tuples/coordinates representing the nodes in a path from start to the goal region
            self.V (set<(int,int)>): A set of Vertices (coordinates) of nodes in the tree
            self.E (set<(int,int),(int,int)>): A set of Edges connecting one node to another node in the tree
        """
        self.initialise(obstacles, bounds, start_pose, goal_region, object_radius,
                        steer_distance, num_iterations, min_iter, resolution, runForFullIterations)

        # Define start and goal in terms of coordinates. The goal is the centroid of the goal polygon.
        x0, y0 = start_pose
        x1, y1 = goal_region.centroid.coords[0]
        start = (x0, y0)
        goal = (x1, y1)

        # Handle edge case where where the start is already at the goal
        if start == goal:
            path = [start, goal]
            self.V.union([start, goal])
            self.E.union([(start, goal)])
        # There might also be a straight path to goal, consider this case before invoking algorithm
        elif self.isEdgeCollisionFree(start, goal):
            path = [start, goal]
            self.V.union([start, goal])
            self.E.union([(start, goal)])
        # Run the appropriate RRT algorithm according to RRT_Flavour
        else:
            if RRT_Flavour == "RRT":
                path, self.V, self.E = self.RRTSearch()
            elif RRT_Flavour == "RRT*":
                path, self.V, self.E = self.RRTStarSearch()
            else:
                # The RRT flavour has no defined algorithm, therefore return None for all values
                return None, None, None

        return self.add_point(path), self.V, self.E

    def RRTSearch(self):
        """Returns path using RRT algorithm.
        Builds a tree exploring from the start node until it reaches the goal region. It works by sampling random points in the map and connecting them with
        the tree we build off on each iteration of the algorithm.
        Returns:
            path (list<(int,int)>): A list of tuples/coordinates representing the nodes in a path from start to the goal region
            self.V (set<(int,int)>): A set of Vertices (coordinates) of nodes in the tree
            self.E (set<(int,int),(int,int)>): A set of Edges connecting one node to another node in the tree
        """

        # Initialize path and tree to be empty.
        path = []
        path_length = float('inf')
        tree_size = 0
        path_size = 0
        self.V.add(self.start_pose)
        goal_centroid = self.get_centroid(self.goal_region)

        # Iteratively sample N random points in environment to build tree
        for i in range(self.N):
            # Change to a value under 1 to bias search towards goal, right now this line doesn't run
            if (random.random() >= 0.995):
                random_point = goal_centroid
            else:
                random_point = self.get_collision_free_random_point()

            # The new point to be added to the tree is not the sampled point, but a colinear point with it and the nearest point in the tree.
            # This keeps the branches short
            nearest_point = self.find_nearest_point(random_point)
            new_point = self.steer(nearest_point, random_point)

            # If there is no obstacle between nearest point and sampled point, add the new point to the tree.
            if self.isEdgeCollisionFree(nearest_point, new_point):
                self.V.add(new_point)
                self.E.add((nearest_point, new_point))
                self.setParent(nearest_point, new_point)
                # If new point of the tree is at the goal region, we can find a path in the tree from start node to goal.
                if self.isAtGoalRegion(new_point):
                    print('PPPPPPPPPPOint at goooooooooooooooooal')
                    # If not running for full iterations, terminate as soon as a path is found.
                    if not self.runForFullIterations and i > self.min_iter:
                        path, tree_size, path_size, path_length = self.find_path(
                            self.start_pose, new_point)
                        break
                    else:  # If running for full iterations, we return the shortest path found.
                        tmp_path, tmp_tree_size, tmp_path_size, tmp_path_length = self.find_path(
                            self.start_pose, new_point)
                        if tmp_path_length < path_length:
                            path_length = tmp_path_length
                            path = tmp_path
                            tree_size = tmp_tree_size
                            path_size = tmp_path_size

        # If no path is found, then path would be an empty list.
        return path, self.V, self.E

    def RRTStarSearch(self):
        """Returns path using RRTStar algorithm.
        Uses the same structure as RRTSearch, except there's an additional 'rewire' call when adding nodes to the tree.
        This can be seen as a way to optimise the branches of the subtree where the new node is being added.
        Returns:
            path (list<(int,int)>): A list of tuples/coordinates representing the nodes in a path from start to the goal region
            self.V (set<(int,int)>): A set of Vertices (coordinates) of nodes in the tree
            self.E (set<(int,int),(int,int)>): A set of Edges connecting one node to another node in the tree
        """
        # Code is very similar to RRTSearch, so for simplicity's sake only the main differences have been commented.
        path = []
        path_length = float('inf')
        tree_size = 0
        path_size = 0
        self.V.add(self.start_pose)
        goal_centroid = self.get_centroid(self.goal_region)

        for i in range(self.N):
            if i%100 == 0: print(i)
            if i>2000: self.obj_radius=1.8
            if i>6000: self.obj_radius=1.7

            if (random.random() >= 0.995) and not self.path_found:
                print('random_point = goal_centroid', i)
                self.random = True
                print(goal_centroid)
                random_point = goal_centroid
            else:
                self.random = False
                random_point = self.get_collision_free_random_point()

            nearest_point = self.find_nearest_point(random_point)
            new_point = self.steer(nearest_point, random_point)

            if self.isEdgeCollisionFree(nearest_point, new_point):
                if self.random: print('isEdgeCollisionFree')
                # Find the nearest set of points around the new point
                nearest_set = self.find_nearest_set(new_point)
                min_point = self.find_min_point(
                    nearest_set, nearest_point, new_point)
                self.V.add(new_point)
                self.E.add((min_point, new_point))
                self.setParent(min_point, new_point)
                # if self.random:
                #     print('min_point', min_point, 'new_point', new_point)
                #     print('setParent')
                # Main difference between RRT and RRT*, modify the points in the nearest set to optimise local path costs.
                self.rewire(nearest_set, min_point, new_point)
                # if self.random: print('rewire')
                if self.isAtGoalRegion(new_point):
                    self.path_found = True
                    print('path_found', i)
                    # print('self.min_iter', self.min_iter)
                    if not self.runForFullIterations and i > self.min_iter:
                        path, tree_size, path_size, path_length = self.find_path(
                            self.start_pose, new_point)
                        break
                    else:
                        tmp_path, tmp_tree_size, tmp_path_size, tmp_path_length = self.find_path(
                            self.start_pose, new_point)
                        if tmp_path_length < path_length:
                            path_length = tmp_path_length
                            path = tmp_path
                            tree_size = tmp_tree_size
                            path_size = tmp_path_size

            if self.path_found and i > self.min_iter:
                break
        return path, self.V, self.E

    # def replanning(self, new_bounds, init_pose, other_car):
    #     print('init_pose', init_pose)
    #     self.obstacles.append(other_car)
    #     # self.ridof_bad_vertices()
    #     self.path_found = False
    #     # self.bounds = new_bounds
    #     # self.min_iter = 500
    #     self.start_pose = init_pose
    #     # self.replan = True
    #     self.obj_radius = 2
    #     self.V = set()
    #     self.E = set()
    #     self.child_to_parent_dict = dict()  # key = child, value = parent

    #     path, self.V, self.E = self.RRTStarSearch()
    #     return path, self.V, self.
    
    def replanning(self, pt, remaining_path, psi, face_car, other_cars):
        for cars in other_cars:
            self.obstacles.append(cars)
        
        face_car_centroid = face_car.centroid.coords.xy[0][0], face_car.centroid.coords.xy[1][0]

        miny, maxy = self.get_min_max_y(pt, psi, face_car)
        #try to the left
        k=1
        traj = self.bypass(pt, face_car_centroid, k, miny, maxy, psi)
        if self.is_traj_free(traj, remaining_path):
            traj = traj+remaining_path  #concatenation
            return traj
        #try to the right
        k=-1
        traj = self.bypass(pt, face_car_centroid, k, miny, maxy, psi)
        if self.is_traj_free(traj, remaining_path):
            traj = traj+remaining_path  #concatenation
            return traj
        return None


    """
    ******************************************************************************************************************************************
    ***************************************************** Helper Functions *******************************************************************
    ******************************************************************************************************************************************
    """

    def sample(self, c_max, c_min, x_center, C):
        if c_max < float('inf'):
            r = [c_max / 2.0, math.sqrt(c_max**2 - c_min**2) /
                 2.0, math.sqrt(c_max**2 - c_min**2)/2.0]
            L = np.diag(r)
            x_ball = self.sample_unit_ball()
            random_point = np.dot(np.dot(C, L), x_ball) + x_center
            random_point = (random_point[(0, 0)], random_point[(1, 0)])
        else:
            random_point = self.get_collision_free_random_point()
        return random_point

    def sample_unit_ball(self):
        a = random.random()
        b = random.random()

        if b < a:
            tmp = b
            b = a
            a = tmp
        sample = (b*math.cos(2*math.pi*a/b), b*math.sin(2*math.pi*a/b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def find_nearest_set(self, new_point):
        points = set()
        ball_radius = self.find_ball_radius()
        for vertex in self.V:
            euc_dist = self.euclidian_dist(new_point, vertex)
            if euc_dist < ball_radius:
                points.add(vertex)
        return points

    def find_ball_radius(self):
        unit_ball_volume = math.pi
        n = len(self.V)
        dimensions = 2.0
        gamma = (2**dimensions)*(1.0 + 1.0/dimensions) * \
            (self.maxx - self.minx) * (self.maxy - self.miny)
        ball_radius = min(((gamma/unit_ball_volume) * math.log(n) / n)
                          ** (1.0/dimensions), self.steer_distance)
        return ball_radius

    def find_min_point(self, nearest_set, nearest_point, new_point):
        min_point = nearest_point
        min_cost = self.cost(nearest_point) + \
            self.linecost(nearest_point, new_point)
        for vertex in nearest_set:
            if self.isEdgeCollisionFree(vertex, new_point):
                temp_cost = self.cost(vertex) + \
                    self.linecost(vertex, new_point)
                if temp_cost < min_cost:
                    min_point = vertex
                    min_cost = temp_cost
        return min_point

    def rewire(self, nearest_set, min_point, new_point):
        # Discards edges in the nearest_set that lead to a longer path than going through the new_point first
        # Then add an edge from new_point to the vertex in question and update its parent accordingly.
        # print('nearest_set', nearest_set, 'set([min_point])', set([min_point]))
        # print('nearest_set - set([min_point])',nearest_set - set([min_point]))
        for vertex in nearest_set - set([min_point]):
            if self.isEdgeCollisionFree(vertex, new_point):
                # if self.random: print('isEdgeCollisionFree')
                # if self.random: print('self.cost(vertex)', self.cost(vertex))
                # if self.random: print('self.cost(new_point)', self.cost(new_point))
                # if self.random: print('self.linecost(vertex, new_point)', self.linecost(vertex, new_point))
                if self.cost(vertex) > self.cost(new_point) + self.linecost(vertex, new_point):
                    parent_point = self.getParent(vertex)
                    self.E.discard((parent_point, vertex))
                    self.E.discard((vertex, parent_point))
                    self.E.add((new_point, vertex))
                    self.setParent(new_point, vertex)

    def cost(self, vertex):
        path, tree_size, path_size, path_length = self.find_path(
            self.start_pose, vertex)
        return path_length

    def linecost(self, point1, point2):
        return self.euclidian_dist(point1, point2)

    def getParent(self, vertex):
        # if self.replan: print('vertex', vertex)
        return self.child_to_parent_dict[vertex]

    def setParent(self, parent, child):
        self.child_to_parent_dict[child] = parent

    def get_random_point(self):
        x = self.minx + random.random() * (self.maxx - self.minx)
        y = self.miny + random.random() * (self.maxy - self.miny)
        return (x, y)

    def get_collision_free_random_point(self):
        # Run until a valid point is found
        while True:
            point = self.get_random_point()
            # print(point)
            # Pick a point, if no obstacle overlaps with a circle centered at point with some obj_radius then return said point.
            buffered_point = Point(point).buffer(
                self.obj_radius, self.resolution)
            if self.isPointCollisionFree(buffered_point):
                return point

    def isPointCollisionFree(self, point):
        for obstacle in self.obstacles:
            if obstacle.contains(point):
                return False

        # list_query = self.tree.query(point)
        # if len(list_query) > 0:
        #     for poly in list_query:
        #         if point.intersects(poly): return False
        return True

    def find_nearest_point(self, random_point):
        closest_point = None
        min_dist = float('inf')
        for vertex in self.V:
            euc_dist = self.euclidian_dist(random_point, vertex)
            if euc_dist < min_dist:
                min_dist = euc_dist
                closest_point = vertex
        return closest_point

    def isOutOfBounds(self, point):
        if ((point[0] - self.obj_radius) < self.minx):
            return True
        if ((point[1] - self.obj_radius) < self.miny):
            return True
        if ((point[0] + self.obj_radius) > self.maxx):
            return True
        if ((point[1] + self.obj_radius) > self.maxy):
            return True
        return False

    def isEdgeCollisionFree(self, point1, point2):
        if self.isOutOfBounds(point2):
            return False
        line = LineString([point1, point2])
        expanded_line = line.buffer(self.obj_radius, self.resolution)
        # expanded_pt1 = Point(point1).buffer(self.obj_radius+0.5, self.resolution)
        # if not self.isAtGoalRegion(point2):
        #     expanded_pt2 = Point(point2).buffer(self.obj_radius+0.5, self.resolution)
        # else:
        #     expanded_pt2 = Point(point2)
        for i, obstacle in enumerate(self.obstacles):
            if i==len(self.obstacles)-1:    #last element => dont buffer the line bc bugs with goal region
                if line.intersects(obstacle):
                    if self.isAtGoalRegion(point2): print('intersect last obstacle')
                    return False
            else:
                if expanded_line.intersects(obstacle):# or not self.isPointCollisionFree(expanded_pt1) or not self.isPointCollisionFree(expanded_pt2):
                    if self.isAtGoalRegion(point2): print('intersect normal obstacle')
                    return False

        # if not self.isPointCollisionFree(expanded_pt1) or not self.isPointCollisionFree(expanded_pt2):
        #     if self.random: print('intersect')
        #     return False
        # list_query = self.tree.query(expanded_line)
        # if len(list_query) > 0:
        #     for poly in list_query:
        #         if expanded_line.intersects(poly):
        #             if self.random: print('intersect')
        #             return False
        return True

    def steer(self, from_point, to_point):
        fromPoint_buffered = Point(from_point).buffer(
            self.obj_radius, self.resolution)
        toPoint_buffered = Point(to_point).buffer(
            self.obj_radius, self.resolution)
        if fromPoint_buffered.distance(toPoint_buffered) < self.steer_distance:
            return to_point
        else:
            from_x, from_y = from_point
            to_x, to_y = to_point
            theta = math.atan2(to_y - from_y, to_x - from_x)
            new_point = (from_x + self.steer_distance * math.cos(theta),
                         from_y + self.steer_distance * math.sin(theta))
            return new_point

    def isAtGoalRegion(self, point):
        # buffered_point = Point(point).buffer(self.obj_radius, self.resolution)
        # intersection = buffered_point.intersection(self.goal_region)
        # inGoal = intersection.area / buffered_point.area
        # if inGoal > 0:
        #     print('IIIIIIIIIIIIIIIIN GOOOOOOOOOOOOOAL', inGoal)
        # return inGoal >= 0.5
        return self.goal_region.contains(Point(point))

    def euclidian_dist(self, point1, point2):
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)

    def find_path(self, start_point, end_point):
        # Returns a path by backtracking through the tree formed by one of the RRT algorithms starting at the end_point until reaching start_node.
        path = [end_point]
        tree_size, path_size, path_length = len(self.V), 1, 0
        current_node = end_point
        previous_node = None
        target_node = start_point
        while current_node != target_node:
            # if self.random: print('self.getParent(current_node)', self.getParent(current_node))
            parent = self.getParent(current_node)
            path.append(parent)
            previous_node = current_node
            current_node = parent
            path_length += self.euclidian_dist(current_node, previous_node)
            path_size += 1
        path.reverse()
        return path, tree_size, path_size, path_length

    def get_centroid(self, region):
        centroid = region.centroid.wkt
        filtered_vals = centroid[centroid.find("(")+1:centroid.find(")")]
        filtered_x = filtered_vals[0:filtered_vals.find(" ")]
        filtered_y = filtered_vals[filtered_vals.find(" ") + 1: -1]
        (x, y) = (float(filtered_x), float(filtered_y))
        return (x, y)

    def add_point(self,path):   #add points in the middle of long lines to avoid the car to deviate from the path
        new_path = [path[0]]
        for i in range(len(path) - 1):
            distance = np.linalg.norm(np.array([path[i][0]-path[i+1][0], path[i][1]-path[i+1][1]]))
            if distance >= 24:
                dir = self.unitvecteur(path[i],path[i+1])
                d = distance/3
                new_point = (path[i][0] + d*dir[0],path[i][1] + d*dir[1])
                new_path.append(new_point)
                new_point = (path[i+1][0] - d*dir[0],path[i+1][1] - d*dir[1])
                new_path.append(new_point)

            elif distance >= 8:
                dir = self.unitvecteur(path[i],path[i+1])
                d = distance/2
                new_point = (path[i][0] + d*dir[0],path[i][1] + d*dir[1])
                new_path.append(new_point)
                
            new_path.append(path[i+1])

        return new_path

    def unitvecteur(self,start,end):

        vector = np.array([end[0]-start[0],end[1]-start[1]])

        return vector/np.linalg.norm(vector)

    def ridof_bad_vertices(self):
        print('ridof_bad_vertices')
        #REMOVE START-POSE BECAUSE IT HAS NO PARENT -> AVOID BUGS
        self.V.remove(self.start_pose)
        new_dict_ = {key: value for key, value in self.child_to_parent_dict.items() if value is not self.start_pose}    #ensure to get rid of the start pose cause it doesn't have parent
        self.child_to_parent_dict = new_dict_
        new_E = {e for e in self.E if (e[0] != self.start_pose and e[1] != self.start_pose)}
        self.E = new_E
        E_ = self.E.copy()
        for e in self.E:
            if not self.isEdgeCollisionFree(e[0],e[1]): #if collision with the edge just remove everything
                print('e isEdgeCollisionFree')
                E_.remove(e)
                if e[1] != self.start_pose: del self.child_to_parent_dict[e[1]]
                new_dict_ = {key: value for key, value in self.child_to_parent_dict.items() if key is not e[0]}
                self.V.remove(e[0])
                self.V.remove(e[1])
                new_dict = {key: value for key, value in new_dict_.items() if value is not e[0]}
                new_dict_ = {key: value for key, value in new_dict.items() if value is not e[1]}
                self.child_to_parent_dict = new_dict_
                new_E = {ee for ee in self.E if (ee[0] != e[0] and ee[1] != e[0])}
                new_E_ = {ee for ee in new_E if (ee[0] != e[1] and ee[1] != e[1])}
                E_ = new_E_
        self.E = E_.copy()
        print('self.child_to_parent_dict', self.child_to_parent_dict)
        print('self.V', self.V)
        print('self.E', self.E)
    
    def bypass(self, pt, face_car_centroid, k, miny, maxy, psi):
        if k==1: #to the left
            pt1 = (pt[0]+ np.sqrt((pt[0]-face_car_centroid[0])**2 + (pt[1]-face_car_centroid[1])**2), pt[1]+maxy+k*3)
        else:   #k=-1 => to the right
            pt1 = (pt[0]+ np.sqrt((pt[0]-face_car_centroid[0])**2 + (pt[1]-face_car_centroid[1])**2), pt[1]+miny+k*3)
        pt1 = (self.rotate(pt,pt1,psi))

        pt2 = (pt[0]+np.sqrt((pt[0]-face_car_centroid[0])**2 + (pt[1]-face_car_centroid[1])**2)+7, pt[1])
        pt2 = (self.rotate(pt,pt2,psi))

        return self.create_trajectory([Point(pt), Point(pt1), Point(pt2)])

    def get_min_max_y(self, pt, psi, face_car):
        new_face_car = affinity.affine_transform(affinity.rotate(face_car, -psi, origin=pt, use_radians=True), [1, 0, 0, 1, -pt[0], -pt[1]])
        _, miny, _, maxy = new_face_car.bounds
        return miny, maxy

    def rotate(self, origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    def create_trajectory(self, waypoints):
        x = []
        y = []
        
        for point in waypoints:
            x.append(point.x)
            y.append(point.y)
        
        tck, *rest = interpolate.splprep([x, y], k=2)
        u = np.linspace(0, 1, num = 10)
        x_smooth, y_smooth = interpolate.splev(u, tck)

        l = len(x_smooth)
        path = [(x_smooth[i], y_smooth[i]) for i in range(len(x_smooth))]
        
        return path

    def is_traj_free(self, traj, remaining_path):
        for i in range(len(traj)-1):
            if not self.isEdgeCollisionFree(traj[i], traj[i+1]):
                return False
        if not self.isEdgeCollisionFree(traj[-1], remaining_path[0]):   #edge between end trajectory and remaing path
            return False
        return True

"""
*    Title: PathTrackingBicycle
*    Author: DongChen06
*    Date: 05.01.2023
*    Availability: https://github.com/DongChen06/PathTrackingBicycle
*    Note that the code is slightly modified in order to be compatible with the project
*
"""


# sampling time
dt = 0.1


class Controller2D(object):
    def __init__(self, waypoints):
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_frame = 0
        self._current_timestamp = 0
        self._current_steer = 0
        self.throttle = 0
        self.brake = 0
        self.steer = 0
        self.steer_rate = 0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        self.e_buffer = []  # deque(maxlen=20)
        self._e = 0
        self.e_buffer_steer = []
        self._e_steer = 0

        # parameters for pid speed controller
        self.K_P = 1*10
        self.K_D = 0.001
        self.K_I = 0.3

        # parameters for pid steer rate controller
        self.K_P_steer = 1*2
        self.K_D_steer = 0.001*1
        self.K_I_steer = 0.3*0.1

    def update_values(self, x, y, yaw, speed, steer):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_steer = steer

    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))
            if dist < min_dist:
                min_dist = dist
                min_idx = i
        if min_idx < len(self._waypoints) - 1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
        if desired_speed <= 9:
            self._desired_speed = desired_speed
        else:
            self._desired_speed = desired_speed - 9

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def update_controls(self):
        # update status
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        waypoints = self._waypoints

        # ==================================
        # LONGITUDINAL CONTROLLER, using PID speed controller
        # ==================================
        self._e = v_desired - v  # v_desired
        self.e_buffer.append(self._e)

        if len(self.e_buffer) >= 2:
            _de = (self.e_buffer[-1] - self.e_buffer[-2]) / dt
            _ie = sum(self.e_buffer) * dt
        else:
            _de = 0.0
            _ie = 0.0

        # self.throttle = np.clip((self.K_P * self._e) + (self.K_D * _de / dt) + (self.K_I * _ie * dt), -1.0, 1.0)
        self.throttle = np.clip(
            (self.K_P * self._e) + (self.K_D * _de / dt) + (self.K_I * _ie * dt), -4, 3)

        # ==================================
        # LATERAL CONTROLLER, using stanley steering controller for lateral control.
        # ==================================
        k_e = 0.3*10
        k_v = 20*1

        # 1. calculate heading error
        yaw_path = np.arctan2(
            waypoints[-1][1] - waypoints[0][1], waypoints[-1][0] - waypoints[0][0])
        yaw_diff = yaw_path - yaw
        if yaw_diff > np.pi:
            yaw_diff -= 2 * np.pi
        if yaw_diff < - np.pi:
            yaw_diff += 2 * np.pi
        # print('yaw_path', yaw_path, '///yaw', yaw, '///yaw_diff', yaw_diff)
        # 2. calculate crosstrack error
        current_xy = np.array([x, y])
        crosstrack_error = np.min(
            np.sum((current_xy - np.array(waypoints)[:, :2]) ** 2, axis=1))

        yaw_cross_track = np.arctan2(y - waypoints[0][1], x - waypoints[0][0])
        yaw_path2ct = yaw_path - yaw_cross_track
        if yaw_path2ct > np.pi:
            yaw_path2ct -= 2 * np.pi
        if yaw_path2ct < - np.pi:
            yaw_path2ct += 2 * np.pi
        if yaw_path2ct > 0:
            crosstrack_error = abs(crosstrack_error)
        else:
            crosstrack_error = - abs(crosstrack_error)

        yaw_diff_crosstrack = np.arctan(k_e * crosstrack_error / (k_v + v))

        # print(crosstrack_error, yaw_diff, yaw_diff_crosstrack)

        # 3. control low
        steer_expect = yaw_diff + yaw_diff_crosstrack
        if steer_expect > np.pi:
            steer_expect -= 2 * np.pi
        if steer_expect < - np.pi:
            steer_expect += 2 * np.pi
        steer_expect = min(1.22, steer_expect)
        steer_expect = max(-1.22, steer_expect)

        # 4. update
        steer_output = steer_expect

        # Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * steer_output
        # Clamp the steering command to valid bounds
        self.steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        # ==================================
        # PID steer rate controller
        # ==================================
        self._e_steer = self.steer - self._current_steer
        self.e_buffer_steer.append(self._e_steer)

        if len(self.e_buffer_steer) >= 2:
            _de_steer = (
                self.e_buffer_steer[-1] - self.e_buffer_steer[-2]) / dt
            _ie_steer = sum(self.e_buffer_steer) * dt
        else:
            _de_steer = 0.0
            _ie_steer = 0.0
        # print('steering rate', (self.K_P_steer * self._e_steer) + (self.K_D_steer * _de_steer / dt) + (self.K_I_steer * _ie_steer * dt))
        self.steer_rate = np.clip((self.K_P_steer * self._e_steer) + (
            self.K_D_steer * _de_steer / dt) + (self.K_I_steer * _ie_steer * dt), -0.5, 0.5)
