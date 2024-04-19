#Author: Bruce Chidley
#This file retrieves Kingston data and writes to a RDDL problem file for the purpose of agent-based disease simulation

#Call this file when generating problem files

import osmnx as ox
import random
import argparse
import csv
import pickle

#LOCATION_TYPES = ["residential", "work", "commercial", "education"]

PLACES_OF_INTEREST = {
    'residential': {'residential', 'house', 'apartments', 'dormitory'},
    'work': {'office', 'commercial', 'industrial', 'retail', 'warehouse'},
    'commercial': {'commercial', 'retail'},
    'education': {'school', 'college', 'university'},
}

city = 'Kingston, Ontario, Canada'

#Sets the centroids for when determining residence/school affiliations
queens_main_centroid = (-76.495362, 44.225724)
queens_west_centroid = (-76.515323, 44.226913)
slc_centroid = (-76.527910, 44.223611)
rmc_centroid = (-76.468120, 44.232918)

#Age brackets - 0 represents age 0-9, 1 represents age 10-19, etc.
age_brackets = [0, 1, 2, 3, 4, 5, 6, 7, 8]
#Probabilities that an agent is in a given age bracket (based on Kingston statistics)
age_probs = [0.092, 0.109, 0.154, 0.135, 0.112, 0.127, 0.125, 0.093, 0.058]

#Possibilities for the number of agents in a given home
home_populations = [1, 2, 3, 4, 5]
#Probabilities associated with each number of agents per home (based on Kingston statistics)
home_probs = [0.329, 0.362, 0.138, 0.113, 0.057]

#Parse arguments
def parse_arguments():

    parser = argparse.ArgumentParser(description="Configure simulation parameters")

    #Residence populations. The number of students in residence at each post-secondary institution (default ratio roughly corresponds to real-life numbers)
    parser.add_argument("--queens_residence_pop", type=int, default=100, help="Enter the Queen's residence population")
    parser.add_argument("--slc_residence_pop", type=int, default=50, help="Enter the SLC residence population")
    parser.add_argument("--rmc_residence_pop", type=int, default=40, help="Enter the RMC residence population")

    #Total post-secondary school populations
    parser.add_argument("--queens_pop", type=int, default=250, help="Enter the total Queen's population")
    parser.add_argument("--slc_pop", type=int, default=60, help="Enter the total SLC population")
    parser.add_argument("--rmc_pop", type=int, default=40, help="Enter the total RMC population")

    #Kingston population. Roughly the total number of agents that will be in the simulation (can be up to 4 more due to home population generation)
    parser.add_argument("--kingston_pop", type=int, default=1000, help="Enter the total population")

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

#Get the buildings
def fetch_buildings(loc):
    """Fetch the buildings from the given location."""
    buildings = ox.features_from_place(loc, tags={"building": True})
    return buildings 

#Get places of interest
def fetch_places_of_interest():

    """Fetch the places of interest from the given location."""
    buildings = fetch_buildings(city)
    places_of_interest = {}
    for key, values in PLACES_OF_INTEREST.items():
        places_of_interest[key] = []
        # iterate over the buildings that have a building type in the values
        for index, building in buildings[buildings["building"].isin(values)].iterrows():
            # Grab the centroid for the place of interest
            centroid = building.geometry.centroid
            # Add the name and centroid to the list
            places_of_interest[key].append([building["building"], [centroid.x, centroid.y]])
    return places_of_interest

#Organize all locations so that they are usable for the RDDL problem file creation
#Takes the list of locations output by fetch_places_of_interest(), and the Python file arguments
#Returns a sample of all locations in a dictionary that have the following structure:
#   residential: a list with each element as such -> [type of dwelling, (longitude, latitude), building capacity (only applicable for dorms and apartments), affiliation (queens, slc, rmc, or general), unique tag, distance to closest post-secondary school]
#   education: a list with each element as such -> [type of school, (longitude, latitude), affiliation (queens, slc, rmc, or nothing if regular school), unique tag]
#   work: a list with each element as such -> [type of workplace, (longitude, latitude), unique tag]
#   commercial: a list with each element as such -> [type of commercial building, (longitude, latitude), unique tag]
def organize_locs(all_locs, args):

    #Initializing the housing lists
    queens_res_list = []
    slc_res_list = []
    rmc_res_list = []
    apartment_list = []
    other_home_list = []

    #Will count the number of residential buildings for unique tag assignment (currently unused, but might be useful later)
    home_counter = 0

    for item in all_locs['residential']:

        #Calculate the euclidian distance from a dorm to the centre of each campus
        d_to_queens_main = ox.distance.euclidean(item[1][0], item[1][1], queens_main_centroid[0], queens_main_centroid[1])
        d_to_queens_west = ox.distance.euclidean(item[1][0], item[1][1], queens_west_centroid[0], queens_west_centroid[1])
        d_to_slc = ox.distance.euclidean(item[1][0], item[1][1], slc_centroid[0], slc_centroid[1])
        d_to_rmc = ox.distance.euclidean(item[1][0], item[1][1], rmc_centroid[0], rmc_centroid[1])

        #Home is closest to Queen's
        if (d_to_queens_main==min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc) or d_to_queens_west== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):

            #If it is a dorm, then assign the Queen's residence population to it (Will be scaled down later)
            if (item[0] == 'dormitory'):
                item.append(args.queens_residence_pop)
                queens_res_list.append(item)
            
            #If it is a dorm, then assign the Kingston residence population to it (Will be scaled down later)
            elif (item[0] == 'apartments'):
                item.append(args.kingston_pop)
                apartment_list.append(item)

            #Otherwise, just assign 1 (building is a regular house)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("queens")
            item.append(home_counter)
            item.append(d_to_queens_main)

        #Home is closest to SLC
        elif (d_to_slc== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):


            #If it is a dorm, then assign the SLC's residence population to it (Will be scaled down later)
            if (item[0] == 'dormitory'):
                item.append(args.slc_residence_pop)
                slc_res_list.append(item)
            elif (item[0] == 'apartments'):
                item.append(args.kingston_pop)
                apartment_list.append(item)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("slc")
            item.append(home_counter)
            item.append(d_to_slc)

        #Home is closest to RMC
        elif (d_to_rmc== min(d_to_queens_main, d_to_queens_west, d_to_slc, d_to_rmc)):

            #If it is a dorm, then assign the RMC's residence population to it (Will be scaled down later)
            if (item[0] == 'dormitory'):
                item.append(args.rmc_residence_pop)
                rmc_res_list.append(item)
            elif (item[0] == 'apartments'):
                item.append(args.kingston_pop)
                apartment_list.append(item)
            else:
                item.append(1)
                other_home_list.append(item)

            item.append("rmc")
            item.append(home_counter)
            item.append(d_to_rmc)
    
        home_counter += 1

    #Takes a random sample of the dormitory buildings
    #It is done this way to ensure that dormitory buildings for each campus are present in meaningful ways.
    #Simply taking a sample of the entire residential building list could leave out important residences
    queens_res_list = random.sample(queens_res_list, args.queens_residences)
    rmc_res_list = random.sample(rmc_res_list, args.rmc_residences)
    slc_res_list = slc_res_list

    #Leaves room for roughly 20 people per apartment
    apartment_list = random.sample(apartment_list, max(1, min(len(apartment_list), round((args.kingston_pop - args.queens_pop - args.slc_pop - args.rmc_pop) / 20))))
    #Leaves room for 6 people per home
    other_home_list = random.sample(other_home_list, max(1, min(len(other_home_list), round((args.kingston_pop - args.queens_pop - args.slc_pop - args.rmc_pop) / 6))))

    #Assigning buildings capacity according to the user-specified populations
    for item in all_locs['residential']:
        if (item[0] == 'dormitory'):
            if (item[3] == 'queens'):
                item[2] = round(item[2]/len(queens_res_list))
            elif (item[3] == 'slc'):
                item[2] = round(item[2]/len(slc_res_list))
            elif (item[3] == 'rmc'):
                item[2] = round(item[2]/len(rmc_res_list))
        elif (item[0] == 'apartments'):
            item[2] = round((item[2] - args.queens_residence_pop - args.slc_residence_pop - args.rmc_residence_pop - len(other_home_list)) / len(apartment_list))

    #Performing a very similar task here, but with education buildings
    queens_edu_list = []
    slc_edu_list = []
    rmc_edu_list = []
    other_edu_list = []

    #Job counter is used for both education environments and workplaces, and represents the building's unique tag. This is because an agent will either go to school or work, and so we give them unique tags simultaneously
    job_counter = 0
    for item in all_locs['education']:
        if (item[0] == 'college'):
            item.append("slc")
            item.append(job_counter)
            slc_edu_list.append(item)
        
        elif (item[0] == 'university'):
            d_to_queens_main = ox.distance.euclidean(item[1][0], item[1][1], queens_main_centroid[0], queens_main_centroid[1])
            d_to_queens_west = ox.distance.euclidean(item[1][0], item[1][1], queens_west_centroid[0], queens_west_centroid[1])
            d_to_rmc = ox.distance.euclidean(item[1][0], item[1][1], rmc_centroid[0], rmc_centroid[1])

            if (d_to_rmc== min(d_to_rmc, d_to_queens_main, d_to_queens_west)):
                item.append("rmc")
                item.append(job_counter)
                rmc_edu_list.append(item)
            
            else:
                item.append("queens")
                item.append(job_counter)
                queens_edu_list.append(item)
        
        #Otherwise, do not append an affiliation. This is so that the regular schools have the same format as work and stores, making agent assignment straightforward
        else:
            item.append(job_counter)
            other_edu_list.append(item)

        job_counter += 1

    #Workplaces use the same counter
    for item in all_locs['work']:

        item.append(job_counter)
        job_counter += 1
    
    #Stores are given tags in the same way, using a separate counter
    store_counter = 0
    for item in all_locs['commercial']:

        item.append(store_counter)
        store_counter += 1

    #Once again, samples are taken in this way to ensure that post-secondary education buildings are present
    #Roughly 20 students per post-secondary education building, and 50 students per general school
    queens_edu_list = random.sample(queens_edu_list, max(1, min(len(queens_edu_list), round(args.queens_pop / 20))))
    slc_edu_list = random.sample(slc_edu_list, max(1, min(len(slc_edu_list), round(args.slc_pop / 20))))
    rmc_edu_list = random.sample(rmc_edu_list, max(1, min(len(rmc_edu_list), round(args.rmc_pop / 20))))   
    other_edu_list = random.sample(other_edu_list, max(1, min(len(other_edu_list), round((args.kingston_pop - args.queens_pop - args.slc_pop - args.rmc_pop) / 50))))

    #Roughly 15 people per commercial building
    commercial_list = random.sample(all_locs['commercial'], max(1, min(len(all_locs['commercial']), round(args.kingston_pop / 15))))

    #Roughly 20 people per workplace
    work_list = random.sample(all_locs['work'], max(1, min(len(all_locs['work']), round(args.kingston_pop / 20))))

    #Assigning values to the keys according to the random samples generated above
    all_locs['residential'] = queens_res_list + rmc_res_list + slc_res_list + apartment_list + other_home_list
    all_locs['education'] = queens_edu_list + rmc_edu_list + slc_edu_list + other_edu_list
    all_locs['work'] = work_list
    all_locs['commercial'] = commercial_list

    return all_locs

#Takes in all residential buildings, all general schools, all stores, all queens/slc/rmc educational buildings, all workplaces, and the student + general populations 
#Returns a list of items with the following format: [[agent, student status, age bracket, home, job, store], home coordinates, job coordinates, store coordinates]
def assign_agents(all_housing, general_schools, stores, queens_buildings, slc_buildings, rmc_buildings, workplaces, student_pops, general_pop):

    #Will be returned at the end
    agent_homes = []

    #Keeps track of the agents and homes for naming purposes
    agent_counter = 0
    home_counter = 0

    #Each list holds residential buildings that are closest to a given campus
    queens_homes = []
    slc_homes = []
    rmc_homes = []

    for home in all_housing:
        if home[3] == "queens":
            queens_homes.append(home)
        elif home[3] == "slc":
            slc_homes.append(home)
        elif home[3] == "rmc":
            rmc_homes.append(home)

    #Loops through the populations for each school (position 0: Queen's, position 1: SLC, position 2: RMC)
    for i in range (0, len(student_pops)):

        #While there are still more students to be assigned
        while student_pops[i] > 0:

            #Selects the residential building closest to a given campus. Generally selects dorms first
            if (i == 0):
                housing = min(queens_homes, key=lambda x: x[5])
            elif (i == 1):
                housing = min(slc_homes, key=lambda x: x[5])
            else:
                housing = min(rmc_homes, key=lambda x: x[5])

            #The capacity for a given residential building
            cap = housing[2]

            #Loops while a building is still under capacity and students still need to be assigned
            while (cap > 0 and student_pops[i] > 0):

                #Each home can hold 1-4 students
                current_in_house = random.randint(1, 4)

                for j in range (current_in_house):

                    #Assigns the agent a building based on what campus the residential building is nearest to
                    if housing[3] == "queens":
                        job = random.choice(queens_buildings)
                        student_pops[0] -= 1
                    
                    elif housing[3] == "slc":
                        job = random.choice(slc_buildings)
                        student_pops[1] -= 1

                    elif housing[3] == "rmc":
                        job = random.choice(rmc_buildings)
                        student_pops[2] -= 1
                    
                    #Assign agent a store
                    store = assign_store(housing[1], stores, 2)

                    #Appends everything to the agent_homes list
                    agent_homes.append([["a" + str(agent_counter), "Student", 2, housing[len(housing)-2], job[3], store[0][2], store[1][2]], housing[1], job[1], store[0][1], store[1][1]])

                    #Increases the agent #, and decreases the capacity
                    cap -= 1
                    agent_counter += 1

                #Changes the unique home value
                home_counter += 1

            #For each campus option, if the capacity is less than 0, remove that residential building from the list. Otherwise, change its capacity to whatever the current capacity is
            #This is so that apartments can be partially filled by students, with the remaining units possibly being filled up by other people in the next section
            if (i == 0):
                if (cap <= 0):
                    queens_homes.remove(min(queens_homes, key=lambda x: x[5]))
                else:
                    min(queens_homes, key=lambda x: x[5])[2] = cap
            elif (i == 1):
                if (cap <= 0):
                    slc_homes.remove(min(slc_homes, key=lambda x: x[5]))
                else:
                    min(slc_homes, key=lambda x: x[5])[2] = cap
            else:
                if (cap <= 0):
                    rmc_homes.remove(min(rmc_homes, key=lambda x: x[5]))
                else:
                    min(rmc_homes, key=lambda x: x[5])[2] = cap
            
    #Combine these residential buildings back into one list, after deleting items and changing capacity values
    all_homes = queens_homes + slc_homes + rmc_homes

    #Remove dorms for the next section
    for item in all_homes:
        if item[0] == 'dormitory':
            all_homes.remove(item)

    random.shuffle(all_homes)

    #Loops while there are still people left to assign
    while general_pop > 0:
        
        #Choose a random home from the list of all homes
        home = random.choice(all_homes)

        cap = home[2]

        #Loops while a building is still under capacity and people still need to be assigned
        while (cap > 0 and general_pop > 0):

            #Each home can hold 1-5 people based on distribution stats
            current_in_house = random.choices(population=home_populations, weights=home_probs)[0]

            for j in range (current_in_house):

                age = random.choices(population=age_brackets, weights=age_probs)[0]

                if (age <= 1):
                    job = random.choice(general_schools)
                elif ((age >= 2) and (age <= 6)):
                    #Assigns the agent a workplace
                    job = random.choice(workplaces)
                else:
                    job = assign_store(home[1], stores, 1)[0]
                general_pop -= 1

                #Assign agent a store
                store = assign_store(home[1], stores, 2)
                
                #Appends everything to the agent_homes list
                agent_homes.append([["a" + str(agent_counter), "other", age, home[len(home)-2], job[2], store[0][2], store[1][2]], home[1], job[1], store[0][1], store[1][1]])

                #Increases the agent #, and agents currently occupying a building
                cap -= 1
                agent_counter += 1

            #Changes the unique home value
            home_counter += 1

        #Home always removed. Even if a building is not totally occupied, no more agents will be assigned anyways, so it can just be removed
        all_homes.remove(home)

    return agent_homes

#Assigns agents stores to visit on the weekend
def assign_store(home_coords, stores_list, num_stores):

    temp = stores_list.copy()

    store_return = []

    #Loops through all stores to find which one is the closest to the input home coordinates
    for i in range(0, num_stores):
        min_d_to_store = ox.distance.euclidean(home_coords[0], home_coords[1], temp[0][1][0], temp[0][1][1])
        coords = temp[0][1]
        assignment = temp[0][2]
        closest_store = temp[0]

        for current_store in temp:
            d_to_store = ox.distance.euclidean(home_coords[0], home_coords[1], current_store[1][0], current_store[1][1])
            if d_to_store <= min_d_to_store:
                closest_store = current_store
                min_d_to_store = d_to_store
                type = current_store[0]
                coords = current_store[1]
                assignment = current_store[2]

        store_return.append([type, coords, assignment])

        temp.remove(closest_store)

    return store_return


def create_loc_agent(locs_to_be_assigned, agents_to_assign):

    loc_dict = {}

    for loc in locs_to_be_assigned['residential']:
        
        loc_dict[loc[4]] = ('home', [], loc[1])

    for loc in locs_to_be_assigned['work']:

        loc_dict[loc[2]] = ('job', [], loc[1])

    for loc in locs_to_be_assigned['education']:
        
        loc_dict[loc[len(loc)-1]] = (('job', [], loc[1]))

    for loc in locs_to_be_assigned['commercial']:

        loc_dict[loc[2]] = ('store', [], loc[1])

    
    for agent in agents_to_assign:

        loc_dict[agent[0][3]][1].append(agent[0][0])
        loc_dict[agent[0][4]][1].append(agent[0][0])
        loc_dict[agent[0][5]][1].append(agent[0][0])

    
    loc_list_final = []

    for key in loc_dict:

        loc_list_final.append([key, loc_dict[key][0], loc_dict[key][1], loc_dict[key][2]])


    return loc_list_final
    



#Writes info to RDDL problem file
#Takes the list of all agents with all assigned information, and the passed arguments
def write_to_RDDL(agent_complete, args):

    with open('problem.rddl', 'w') as f:
        f.write ('non-fluents covid-sim_nf_1 {')
        f.write('\n')
        f.write('\tdomain = covid-sim;')
        f.write('\n')
        f.write('\tobjects {')
        f.write('\n')

        agent_string = 'agent : {'
        for agent in agent_complete:
            agent_string = agent_string + agent[0][0] + ', '
        agent_string = agent_string[:-2] + '};'

        f.write('\t\t' + agent_string)

        f.write('\n')
        f.write('\t};')
        f.write('\n')
        f.write('\tnon-fluents {')
        f.write('\n')
        f.write('\n')

        f.write('\t\tPENALTY_MASK_MANDATE_ALL = ' + str(args.mask_penalty_all) + ';')
        f.write('\n')
        f.write('\t\tPENALTY_VACCINE_MANDATE_ALL = ' + str(args.vaccine_penalty_all) + ';')
        f.write('\n')
        f.write('\t\tPENALTY_MASK_MANDATE_STUDENTS = ' + str(args.mask_penalty_students) + ';')
        f.write('\n')
        f.write('\t\tPENALTY_VACCINE_MANDATE_STUDENTS = ' + str(args.vaccine_penalty_students) + ';')
        f.write('\n')
        f.write('\t\tPENALTY_NON_ICU_CAPACITY = ' + str(args.non_icu_penalty) + ';')
        f.write('\n')
        f.write('\t\tPENALTY_ICU_CAPACITY = ' + str(args.icu_penalty) + ';')
        f.write('\n')
        f.write('\n')

        f.write('\t\tMASK_FACTOR = ' + str(args.mask_factor) + ';')
        f.write('\n')
        f.write('\t\tVACCINE_FACTOR = ' + str(args.vaccine_factor) + ';')
        f.write('\n')
        f.write('\t\tMASK_CHANCE = ' + str(args.mask_chance) + ';')
        f.write('\n')
        f.write('\t\tHOSPITAL_BEDS_NON_ICU = ' + str(args.non_icu_beds) + ';')
        f.write('\n')
        f.write('\t\tHOSPITAL_BEDS_ICU = ' + str(args.icu_beds) + ';')
        f.write('\n')
        f.write('\n')


        for agent in agent_complete:

            if (agent[0][1] == 'Student'):
                f.write('\t\tAGENT_STUDENT(' + agent[0][0] + ') = true;')
            else:
                f.write('\t\tAGENT_STUDENT(' + agent[0][0] + ') = false;')
            f.write('\n')
            f.write('\t\tAGENT_AGE(' + agent[0][0] + ') = ' + str(agent[0][2]) + ';')
            f.write('\n')
            f.write('\t\tAGENT_HOME(' + agent[0][0] + ') = ' + str(agent[0][3]) + ';')
            f.write('\n')
            f.write('\t\tAGENT_JOB(' + agent[0][0] + ') = ' + str(agent[0][4]) + ';')
            f.write('\n')
            f.write('\t\tAGENT_STORE_1(' + agent[0][0] + ') = ' + str(agent[0][5]) + ';')
            f.write('\n')
            f.write('\t\tAGENT_STORE_2(' + agent[0][0] + ') = ' + str(agent[0][6]) + ';')
            f.write('\n')

            
        
        f.write('\t};')
        f.write('\n')
        f.write('}')
        f.write('\n')
        f.write('\n')

        f.write('instance covid-sim_inst_1 {')
        f.write('\n')
        f.write('\tdomain = covid-sim;')
        f.write('\n')
        f.write('\tnon-fluents = covid-sim_nf_1;')
        f.write('\n')
        f.write('\tinit-state {')
        f.write('\n')

        for agent in agent_complete:
            infectious = random.randint(1,10)

            if infectious == 1:
                f.write('\t\tinfectious(' + agent[0][0] + ');')
                f.write('\n')
                f.write('\t\tinfectious_total_time(' + agent[0][0] + ') = 16;')
                f.write('\n')
            
        f.write('\t};')
        f.write('\n')
        f.write('\n')
        f.write('\thorizon = ' + str(args.horizon) + ';')
        f.write('\n')
        f.write('\tdiscount = 1.0;')
        f.write('\n')
        f.write('}')

def main_kingston_geo():

    args = parse_arguments()

    #Collects locations via the OSMnx package
    locs = fetch_places_of_interest()

    #Organizes the locations and performs some population and distance calcualtions
    organized_locs = organize_locs(locs, args)

    #Retrieves all education buildings of different types
    all_schools = []
    queens = []
    slc = []
    rmc = []

    #Separates the types of schools for ease of processing
    for item in organized_locs['education']:
        if (item[0] == 'school'):
            all_schools.append(item)
        elif (item[2] == 'queens'):
            queens.append(item)
        elif (item[2] == 'slc'):
            slc.append(item)
        elif (item[2] == 'rmc'):
            rmc.append(item)

    #Assign school + general populations
    school_populations = [args.queens_residence_pop, args.slc_residence_pop, args.rmc_residence_pop]
    general_population = args.kingston_pop - args.queens_residence_pop - args.slc_residence_pop - args.rmc_residence_pop

    #Calls the functions
    agent_complete = assign_agents(organized_locs['residential'], all_schools, organized_locs['commercial'], queens, slc, rmc, organized_locs['work'], school_populations, general_population)

    loc_with_agents = create_loc_agent(organized_locs, agent_complete)

    print(len(loc_with_agents))
    print(len(agent_complete))
    print(len(organized_locs))

    filehandler = open("loc_with_agents.p","wb")
    pickle.dump(loc_with_agents,filehandler)
    filehandler.close()

    filehandler = open("agent_complete.p","wb")
    pickle.dump(agent_complete,filehandler)
    filehandler.close()

    filehandler = open("organized_locs.p","wb")
    pickle.dump(organized_locs,filehandler)
    filehandler.close()

    return (loc_with_agents, agent_complete, organized_locs)

    #with open('Locations.csv', 'w', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow(loc_with_agents) # Use writerow for single list



def shuffle_locations_uniform(passed_organized_locs):

    args = parse_arguments()

    longitudes = []
    latitudes = []

    for type_building in passed_organized_locs.values():

        for item in type_building:
            longitudes.append(float(item[1][0]))
            latitudes.append(float(item[1][1]))

    max_longitude = max(longitudes)
    min_longitude = min(longitudes)
    max_latitude = max(latitudes)
    min_latitude = min(latitudes)

    for type_building in passed_organized_locs.values():

        for item in type_building:

            if item[0] != 'dormitory':

                new_longitude = random.uniform(min_longitude, max_longitude)
                new_latitude = random.uniform(min_latitude, max_latitude)

                item[1][0] = new_longitude
                item[1][1] = new_latitude


    #Retrieves all education buildings of different types
    all_schools = []
    queens = []
    slc = []
    rmc = []

    #Separates the types of schools for ease of processing
    for item in passed_organized_locs['education']:
        if (item[0] == 'school'):
            all_schools.append(item)

    #Assign school + general populations
    school_populations = []
    general_population = args.kingston_pop

    #Calls the functions
    agent_complete = assign_agents(passed_organized_locs['residential'], all_schools, passed_organized_locs['commercial'], queens, slc, rmc, passed_organized_locs['work'], school_populations, general_population)

    loc_with_agents = create_loc_agent(passed_organized_locs, agent_complete)

    return (loc_with_agents, agent_complete, passed_organized_locs)

if __name__ == "__main__":

    main_kingston_geo()