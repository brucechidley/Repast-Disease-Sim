import pickle

file_1 = open('loc_with_agents.p', 'rb')
location_details = list(pickle.load(file_1))
file_1.close()

file_2 = open('agent_complete.p', 'rb')
agent_details = list(pickle.load(file_2))
file_2.close()

file_3 = open('organized_locs.p', 'rb')
organized_locs = list(pickle.load(file_3))
file_3.close()

for item in agent_details:
    print(item)