import sys
import math
import numpy as np
from typing import Dict, Tuple
from mpi4py import MPI
from dataclasses import dataclass

import numba
from numba import int32, int64
from numba.experimental import jitclass

from repast4py import core, space, schedule, logging, random
from repast4py import context as ctx
from repast4py.parameters import create_args_parser, init_params

from repast4py.space import ContinuousPoint as cpt
from repast4py.space import DiscretePoint as dpt
from repast4py.space import BorderType, OccupancyType

import matplotlib.pyplot as plt

import Kingston_Info
import argparse

import pickle

def parse_arguments():

    parser = argparse.ArgumentParser(description="Configure simulation parameters")

    #Residence populations. The number of students in residence at each post-secondary institution (default ratio roughly corresponds to real-life numbers)
    parser.add_argument("--queens_residence_pop", type=int, default=10, help="Enter the Queen's residence population")
    parser.add_argument("--slc_residence_pop", type=int, default=5, help="Enter the SLC residence population")
    parser.add_argument("--rmc_residence_pop", type=int, default=4, help="Enter the RMC residence population")

    #Total post-secondary school populations
    parser.add_argument("--queens_pop", type=int, default=25, help="Enter the total Queen's population")
    parser.add_argument("--slc_pop", type=int, default=6, help="Enter the total SLC population")
    parser.add_argument("--rmc_pop", type=int, default=4, help="Enter the total RMC population")

    #Kingston population. Roughly the total number of agents that will be in the simulation (can be up to 4 more due to home population generation)
    parser.add_argument("--kingston_pop", type=int, default=100, help="Enter the total population")

    #Number of residences at each post-secondary institution (except for SLC, which only has 1 residence)
    parser.add_argument("--queens_residences", type=int, default=10, help="Enter the number of Queen's residences - max of 30")
    parser.add_argument("--rmc_residences", type=int, default=3, help="Enter the number of RMC residences - max of 7")

    #Penalties associated with the simulation and planner actions
    parser.add_argument("--mask_penalty_all", type=float, default=-10, help="Enter the mask penalty factor for all agents")
    parser.add_argument("--vaccine_penalty_all", type=float, default=-10, help="Enter the vaccine penalty factor for all agents")
    parser.add_argument("--mask_penalty_students", type=float, default=-5, help="Enter the mask penalty factor for students")
    parser.add_argument("--vaccine_penalty_students", type=float, default=-5, help="Enter the vaccine penalty factor for students")
    parser.add_argument("--non_icu_penalty", type=float, default=-8000, help="Enter the non-ICU penalty factor")
    parser.add_argument("--icu_penalty", type=float, default=-8000, help="Enter the ICU penalty factor")

    #The factors that will be multiplied with transmission chance
    parser.add_argument("--mask_factor", type=float, default=0.8, help="Enter the factor that wearing a mask multiplies transmission rate by")
    parser.add_argument("--vaccine_factor", type=float, default=0.4, help="Enter the factor that being vaccinated multiplies transmission rate by")

    #The chance an agent wears a mask
    parser.add_argument("--mask_chance", type=float, default=0.7, help="Enter the chance that an agent wears a mask")

    #The total number of non-ICU and ICU beds
    parser.add_argument("--non_icu_beds", type=int, default=2, help="Enter the total number of non-ICU beds")
    parser.add_argument("--icu_beds", type=int, default=1, help="Enter the total number of ICU beds")

    #The number of time steps for the simulation
    parser.add_argument("--horizon", type=int, default=100, help="Enter the desired number of time steps (horizon)")
    
    #Defines the way the simulation is run
    parser.add_argument("--mode", type=str, default="Init", help="Enter the desired mode (Init if you are creating new problem files, Test if you are drawing from existing problem files)")
    parser.add_argument("--iters", type=int, default=20, help="Enter the number of iterations you wish to run")
    parser.add_argument("--trials", type=int, default=20, help="Enter the number of trials per iteration you wish to run")

    return parser.parse_args()

@dataclass
class MeetLog:
    susceptible_count: int = 0
    exposed_count: int = 0
    infectious_count: int = 0
    recovered_count: int = 0
    total_count: int = 0


class Person(core.Agent):

    TYPE = 0

    def __init__(self, local_id: int, rank: int, agent_info):
        super().__init__(id=local_id, type=Person.TYPE, rank=rank)

        self.info = agent_info

        self.real_coords = [self.info[1], self.info[2], self.info[3], self.info[4]]

        self.home_loc = [self.info[1][0] * (-100000), self.info[1][1] * 100000]
        self.job_loc = [self.info[2][0] * (-100000), self.info[2][1]  * 100000]
        self.store_1_loc = [self.info[3][0] * (-100000), self.info[3][1]  * 100000]
        self.store_2_loc = [self.info[4][0] * (-100000), self.info[4][1]  * 100000]


        self.age_bracket = agent_info[2]
        self.isolating = False

        #Assigns them as infectious from the start depending on some probability (10% here)
        prob_infectious = np.random.uniform(0,1)
        if (prob_infectious <= 0.1):
            self.susceptible = False
            self.exposed = False
            self.infectious = True
            self.recovered = False
            self.total_time_in_class = np.random.normal(16, 4)
        
        else:
            self.susceptible = True
            self.exposed = False
            self.infectious = False
            self.recovered = False
            self.total_time_in_class = 999

        #Same deal with masking and vaccinating
        prob_masked = np.random.uniform(1,2)
        if (prob_masked <= 0.7):
            self.masked_factor = 0.8
        else:
            self.masked_factor = 1

        prob_vaccinated = np.random.uniform(1,2)

        #Set to 1 by default
        self.vaccinated_factor = 1

        #Vaccination rates based on age bracket
        if self.age_bracket == 0:
            if (prob_vaccinated <= 0.251):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 1:
            if (prob_vaccinated <= 0.771):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 2:
            if (prob_vaccinated <= 0.819):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 3:
            if (prob_vaccinated <= 0.851):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 4:
            if (prob_vaccinated <= 0.883):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 5:
            if (prob_vaccinated <= 0.885):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 6:
            if (prob_vaccinated <= 0.940):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 7:
            if (prob_vaccinated <= 0.982):
                self.vaccinated_factor = 0.4
        elif self.age_bracket == 8:
            if (prob_vaccinated <= 0.990):
                self.vaccinated_factor = 0.4

        self.time_in_class = 0

    def move(self):


        #print(model.space.get_location(self))

        if model.time_of_day == 1:
            model.space.move(self, cpt(self.home_loc[0], self.home_loc[1]))
            #print("We move hhoem :) agent " + str(self.info[0][0]))
        
        else:

            if model.day_of_week <= 10:
                model.space.move(self, cpt(self.job_loc[0], self.job_loc[1]))
                #print("jobbing")
            elif model.day_of_week == 12:
                model.space.move(self, cpt(self.store_1_loc[0], self.store_1_loc[1]))
            else:
                model.space.move(self, cpt(self.store_2_loc[0], self.store_2_loc[1]))

        #print(model.space.get_location(self))
        #print("")
                
    def count_states(self, meet_log: MeetLog):

        if self.susceptible:
            meet_log.susceptible_count += 1
        elif self.exposed:
            meet_log.exposed_count += 1
        elif self.infectious:
            meet_log.infectious_count += 1
        else:
            meet_log.recovered_count += 1

        meet_log.total_count += 1
                
    def save(self):
        return (self.uid,[self.info, self.real_coords, self.home_loc, self.job_loc, self.store_1_loc, self.store_2_loc, self.age_bracket, self.isolating,
                          self.susceptible, self.exposed, self.infectious, self.recovered, self.masked_factor, self.vaccinated_factor, self.time_in_class], [])
    
    
    def update(self, data: bool):
        """Updates the state of this agent when it is a ghost
        agent on some rank other than its local one.

        Args:
            data: the new agent state (received_rumor)
        """

        print("here")

        self.info = data[1][0]
        self.real_coords = data[1][1]
        self.home_loc = data[1][2]
        self.job_loc = data[1][3]
        self.store_1_loc = data[1][4]
        self.store_2_loc = data[1][5]
        self.age_bracket = data[1][6]
        self.isolating = data[1][7]
        self.susceptible = data[1][8]
        self.exposed = data[1][9]
        self.infectious = data[1][10]
        self.recovered = data[1][11]
        self.masked_factor = data[1][12]
        self.vaccinated_factor = data[1][13]
        self.time_in_class = data[1][14]
    
    

class Location(core.Agent):

    TYPE = 1

    def __init__(self, a_id, rank, loc_info):
        super().__init__(id=a_id, type=Location.TYPE, rank=rank)

        self.real_coords = [loc_info[3][0], loc_info[3][1]]
        self.adjusted_coords = [loc_info[3][0] * (-100000), loc_info[3][1] * 100000]

        self.all_info = loc_info

        self.loc_type = loc_info[1]

        self.agent_count = 0
        self.susceptible_count = 0
        self.exposed_count = 0
        self.infectious_count = 0
        self.recovered_count = 0
    
    def update_info(self):

        c_space = model.space

        all_agents_at_loc = list(c_space.get_agents(cpt(self.adjusted_coords[0],self.adjusted_coords[1])))

        all_agents_at_loc.remove(self)

        new_agent_list = []

        for item in all_agents_at_loc:
            if item.uid[1] == Location.TYPE:
                all_agents_at_loc.remove(item)
            else:
                if item.info[0][0] in self.all_info[2]:
                    new_agent_list.append(item)
    

        count = 0
        susceptibles = 0
        exposeds = 0
        infectiouses = 0
        recovereds = 0

        for agent_at_loc in new_agent_list:

            count += 1

            if agent_at_loc.susceptible:
                susceptibles += 1
            elif agent_at_loc.exposed:
                exposeds += 1
            elif agent_at_loc.infectious:
                infectiouses += 1
            else:
                recovereds += 1

        self.agent_count = count
        self.susceptible_count = susceptibles
        self.exposed_count = exposeds
        self.infectious_count = infectiouses
        self.recovered_count = recovereds


        #print(str(len(new_agent_list)) + ": " + str(new_agent_list) + ", and " + str(self.all_info))

    def update_agent(self, current_agent, all_agents):

        newly_exposed = False

        if current_agent.susceptible:

            for agent_at_same_loc in all_agents:

                if ((agent_at_same_loc.infectious) and (not agent_at_same_loc.isolating)):

                    odds = (agent_at_same_loc.masked_factor * current_agent.masked_factor) * (current_agent.vaccinated_factor) * (3.32 / 16) / (self.susceptible_count)

                    against = np.random.uniform(0,1)
                    if (against < odds):
                        newly_exposed = True


            if newly_exposed:
                #print("Success!")
                current_agent.susceptible = False
                current_agent.exposed = True
                current_agent.time_in_class = 0

                #Set the time the agent will be in the exposed class
                current_agent.total_time_in_class = np.random.normal(9, 2)

        elif current_agent.exposed and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.exposed = False
            current_agent.infectious = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the infectious class
            current_agent.total_time_in_class = np.random.normal(16, 4)

            #Check to see if agent isolates
            isolating_prob = np.random.uniform(0, 1)
            if isolating_prob <= 0.3:
                current_agent.isolating = True

        #Same for infectious
        elif current_agent.infectious and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.infectious = False
            current_agent.recovered = True
            current_agent.time_in_class = 0

            #Set the time the agent will be in the recovered class
            current_agent.total_time_in_class = 14

            current_agent.isolating = False
        
        #Same for recovered
        elif current_agent.recovered and current_agent.time_in_class >= current_agent.total_time_in_class:
            current_agent.recovered = False
            current_agent.susceptible = True
            current_agent.time_in_class = 0

            #Set to 999, since an agent is susceptible until infected
            current_agent.total_time_in_class = 999


        current_agent.time_in_class += 1

    def step(self):

        c_space = model.space

        all_agents_at_loc = list(c_space.get_agents(cpt(self.adjusted_coords[0],self.adjusted_coords[1])))

        all_agents_at_loc.remove(self)

        new_agent_list = []

        for item in all_agents_at_loc:
            if item.uid[1] == Location.TYPE:
                all_agents_at_loc.remove(item)
            else:
                if item.info[0][0] in self.all_info[2]:
                    new_agent_list.append(item)

        #print("location rank: " + str(self.uid[2]))

        for agent_at_loc in new_agent_list:

            #print(agent_at_loc.uid[2])

            self.update_agent(agent_at_loc, new_agent_list)

    def save(self):

        return (self.uid,[], [self.all_info, self.real_coords, self.adjusted_coords, self.loc_type, self.agent_count, self.susceptible_count, self.exposed_count, self.infectious_count, self.recovered_count])

agent_cache = {}

def restore_agent(agent_data: Tuple):
    """Creates an agent from the specified agent_data.

    This is used to re-create agents when they have moved from one MPI rank to another.
    The tuple returned by the agent's save() method is moved between ranks, and restore_agent
    is called for each tuple in order to create the agent on that rank. Here we also use
    a cache to cache any agents already created on this rank, and only update their state
    rather than creating from scratch.

    Args:
        agent_data: the data to create the agent from. This is the tuple returned from the agent's save() method
                    where the first element is the agent id tuple, and any remaining arguments encapsulate
                    agent state.
    """

    #print(agent_data)

    uid = agent_data[0]
    # 0 is id, 1 is type, 2 is rank

    if uid[1] == Person.TYPE:
        if uid in agent_cache:
            agent_re = agent_cache[uid]
        else:
            agent_re = Person(uid[0], uid[2], agent_data[1][0])
            agent_cache[uid] = agent_re

        # restore the agent state from the agent_data tuple
        agent_re.info = agent_data[1][0]
        agent_re.real_coords = agent_data[1][1]
        agent_re.home_loc = agent_data[1][2]
        agent_re.job_loc = agent_data[1][3]
        agent_re.store_1_loc = agent_data[1][4]
        agent_re.store_2_loc = agent_data[1][5]
        agent_re.age_bracket = agent_data[1][6]
        agent_re.isolating = agent_data[1][7]
        agent_re.susceptible = agent_data[1][8]
        agent_re.exposed = agent_data[1][9]
        agent_re.infectious = agent_data[1][10]
        agent_re.recovered = agent_data[1][11]
        agent_re.masked_factor = agent_data[1][12]
        agent_re.vaccinated_factor = agent_data[1][13]
        agent_re.time_in_class = agent_data[1][14]

        return agent_re

    elif uid[1] == Location.TYPE:
        if uid in agent_cache:
            agent_re = agent_cache[uid]
        else:
            agent_re = Location(uid[0], uid[2], agent_data[2][0])
            agent_cache[uid] = agent_re

        agent_re.all_info = agent_data[2][0]
        agent_re.real_coords = agent_data[2][1]
        agent_re.adjusted_coords = agent_data[2][2]
        agent_re.loc_type = agent_data[2][3]
        agent_re.agent_count = agent_data[2][4]
        agent_re.susceptible_count = agent_data[2][5]
        agent_re.exposed_count = agent_data[2][6]
        agent_re.infectious_count = agent_data[2][7]
        agent_re.recovered_count = agent_data[2][8]

        return agent_re

    

class Model:

    def __init__(self, comm, params, location_info, agents_info):
        self.comm = comm
        self.context = ctx.SharedContext(comm)
        self.rank = self.comm.Get_rank()

        self.runner = schedule.init_schedule_runner(comm)
        self.runner.schedule_repeating_event(1, 1, self.step)
        self.runner.schedule_stop(params['stop.at'])
        self.runner.schedule_end_event(self.at_end)

        box = space.BoundingBox(params['min.longitude'], params['longitude.extent'], params['min.latitude'], params['latitude.extent'], 0, 0)

        self.space = space.SharedCSpace('space', bounds=box, borders=BorderType.Sticky, occupancy=OccupancyType.Multiple,
                                        buffer_size=2, comm=comm, tree_threshold=100)
        self.context.add_projection(self.space)

        world_size = comm.Get_size()

        self.time_of_day = 1
        self.day_of_week = 1

        self.all_location_info = location_info
        self.all_agent_info = agents_info

        self.num_agents = len(agents_info)
        self.num_locations = len(location_info)

        agent_per_rank = int(self.num_agents / world_size)
        #print("world size: " + str(world_size))

        for i in range(agent_per_rank):

            #print("")
            #print(self.rank)
            #print(agents_info[i * (self.rank + 1)])
            #print("")
            h = Person(i + (self.rank * agent_per_rank), self.rank, agents_info[i + (self.rank * agent_per_rank)])

            #print(self.rank, h.info[0][0])

            self.context.add(h)

            x = agents_info[i + (self.rank * agent_per_rank)][1][0] * (-100000)
            y = agents_info[i + (self.rank * agent_per_rank)][1][1] * 100000

            self.space.move(h, cpt(x, y))

        #for person_agent in self.context.agents(Person.TYPE):
        #    print(self.rank, person_agent.info[0][0])
            
        location_per_rank = int(self.num_locations / world_size)

        for i in range(location_per_rank):
            l = Location(i + (self.rank * location_per_rank), self.rank, location_info[i + (self.rank * location_per_rank)])
            #print(self.rank, location_info[i])
            self.context.add(l)

            x = location_info[i + (self.rank * location_per_rank)][3][0] * (-100000)
            y = location_info[i + (self.rank * location_per_rank)][3][1] * 100000

            self.space.move(l, cpt(x,y))

        self.context.synchronize(restore_agent)

        #for person_agent in self.context.agents(Person.TYPE):
        #    print(str(self.rank) + " " + str(person_agent.info[0][0]))

        #print("rank: " + str(self.rank))
        #print(str(min_x) + ", " + str(min_y) + " to " + str(max_x) + ", " + str(max_y))
            
        self.meet_log = MeetLog()
        loggers = logging.create_loggers(self.meet_log, op=MPI.SUM, names={'susceptible_count': 'susceptible'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'exposed_count': 'exposed'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'infectious_count': 'infectious'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'recovered_count': 'recovered'}, rank=self.rank)
        loggers += logging.create_loggers(self.meet_log, op=MPI.SUM, names={'total_count': 'total'}, rank=self.rank)
        self.data_set = logging.ReducingDataSet(loggers, comm, params['meet_log_file'])

        for person_ag in self.context.agents(Person.TYPE):
            person_ag.count_states(self.meet_log)
        self.data_set.log(0)
        self.meet_log.susceptible_count = self.meet_log.exposed_count = self.meet_log.infectious_count = self.meet_log.recovered_count = self.meet_log.total_count = 0

    def run(self):
        self.runner.execute()

    def step(self):

        # print("{}: {}".format(self.rank, len(self.context.local_agents)))

        self.move_agents()

        #for log_ac in self.context.agents(Location.TYPE):
        #    print(self.space.get_location(log_ac), self.rank, log_ac.all_info[0])

        self.context.synchronize(restore_agent)

        self.update_loc()

        self.step_loc()

        self.update_time()

        self.log_all()

        #print(tick, self.rank)
        #print("Susceptible: " + str(self.total_susceptible_count) + ", Exposed: " + str(self.total_exposed_count) + 
        #      ", Infectious: " + str(self.total_infectious_count) + ", Recovered: " + str(self.total_recovered_count))

            
    def move_agents(self):

        for person_agent in self.context.agents(Person.TYPE):
            person_agent.move()

    def update_loc(self):

        for location_agent in self.context.agents(Location.TYPE):

            if ((self.day_of_week % 2 == 1 and location_agent.loc_type == "home") or (self.day_of_week in (2,4,6,8,10) and location_agent.loc_type == "job")
            or (self.day_of_week in (12,14) and location_agent.loc_type == "store")):
                
                location_agent.update_info()

    def step_loc(self):

        for location_agent in self.context.agents(Location.TYPE):

            if ((self.day_of_week % 2 == 1 and location_agent.loc_type == "home") or (self.day_of_week in (2,4,6,8,10) and location_agent.loc_type == "job")
            or (self.day_of_week in (12,14) and location_agent.loc_type == "store")):
                
                location_agent.step()

    def update_time(self):

        if self.time_of_day == 1:
            self.time_of_day = 2
        else:
            self.time_of_day = 1

        if self.day_of_week <= 13:
            self.day_of_week += 1
        else:
            self.day_of_week = 1

    def log_all(self):

        for person_ag in self.context.agents(Person.TYPE):
            person_ag.count_states(self.meet_log)

        tick = self.runner.schedule.tick
        self.data_set.log(tick)
        # clear the meet log counts for the next tick
        self.meet_log.susceptible_count = self.meet_log.exposed_count = self.meet_log.infectious_count = self.meet_log.recovered_count = self.meet_log.total_count = 0

    def at_end(self):
        self.data_set.close()


#info = Kingston_Info.main_kingston_geo()
#location_details = info[0]
#agent_details = info[1]
#organized_locs = info[2]

file_1 = open('loc_with_agents.p', 'rb')
location_details = list(pickle.load(file_1))
file_1.close()

file_2 = open('agent_complete.p', 'rb')
agent_details = list(pickle.load(file_2))
file_2.close()

file_3 = open('organized_locs.p', 'rb')
organized_locs = list(pickle.load(file_3))
file_3.close()

#print(len(location_details), len(agent_details), len(organized_locs))

min_x = 100000000
min_y = 100000000
max_x = 0
max_y = 0

for loc in location_details:

    if loc[3][0] * (-100000)  <= min_x:
        min_x = loc[3][0] * (-100000)
    if loc[3][1] * 100000 <= min_y:
        min_y = loc[3][1] * 100000
    if loc[3][0] * (-100000) >= max_x:
        max_x = loc[3][0] * (-100000)
    if loc[3][1] * 100000 >= max_y:
        max_y = loc[3][1] * 100000

params = {

    "stop.at": 500,
    "min.longitude": int(np.floor(min_x)),
    "longitude.extent": int(np.ceil(max_x)) - int(np.floor(min_x)),
    "min.latitude": int(np.floor(min_y)),
    "latitude.extent": int(np.ceil(max_y)) - int(np.floor(min_y)),
    'meet_log_file': 'output/counts.csv'

}

for i in range (0, 1):
    global model    
    model = Model(MPI.COMM_WORLD, params, location_details, agent_details)
    model.run()

#plt.plot(model.time_steps, model.infectious_counts, color='red', linewidth=0.5)
#plt.plot(model.time_steps, model.susceptible_counts, color='green', linewidth=0.5)
#plt.title('Susceptible/Infectious Agents Over Time')
#plt.xlabel("Time Steps")
#plt.ylabel("Number of Agents")
#plt.legend()

#save_name = "100_agents.png"
#plt.savefig(save_name)
#plt.clf()
