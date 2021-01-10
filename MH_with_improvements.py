import pandas as pd
import random
import math
from random import choice
import numpy as np
import timeit                               #To determine computional time
import gurobipy as gp
from gurobipy import GRB,quicksum
from PH_function_chrom import PH_Chromosome_generator

####################################################################################
#Input_file
File_name = '6M140R026.csv'                         # '6M140.csv' or 'example_3.csv'
####################################################################################

# #Read data
all_data = pd.read_csv(File_name, names = ['Job','Operation','Release_time','Processing_time','Due_date','Tool_set','Tool_set_size'])
data = all_data[6:].reset_index(drop=True) #Including reseting the index so it starts at 0, instead of 6.

# Convert data to right format
data['Job'] = data['Job'].astype(int)
data['Operation'] = data['Operation'].astype(int)
data['Release_time'] = data['Release_time'].astype(float)
data['Processing_time'] = data['Processing_time'].astype(float)
data['Due_date'] = data['Due_date'].astype(float)
data['Tool_set'] = data['Tool_set'].astype(int)
data['Tool_set_size'] = data['Tool_set_size'].astype(int)
data['KEEP_AS_STRING'] = ['String'] * len(data['Job'])

# Define Parameters
O = int(all_data.iloc[0][1])
M = int(all_data.iloc[1][1])
T = int(all_data.iloc[2][1])
C = int(all_data.iloc[3][1])
Set_up_time = 1
Wd = 1                              #(required for objective function). Tardiness weight
Ws = 1                              #(required for objective function). Setup weight
Pu = 0.01                           #probability of uniform mutation
Ps = 0.01                           #probability of swap mutation
Gamma1 = 0.1                        #percentage selected for tournament.
Gamma2 = 0.1                        #required for selecting nr of best parents
NP = 100                            #population size
SE = math.ceil(Gamma2 * NP)              #Elitism
Max_time = 3600
Gc = 20
Beta = 1
n_parameter = 0.50

#Set start time in order to determine computational time
start = timeit.default_timer()

# random seed to get consistent results
#random.seed(42)

#Sets of Jobs, Machines & Toolsets
Jobs = list(int(i) for i in data.Job)
Tool_sets = list(set(int(i) for i in list(data.Tool_set)))
machines = list(set(int(i+1) for i in range(0,M)))
####################################################################################








# =============================================================================
# Interpret solution in the form of (job, operation) instead of (job, machine) for all machines in a dictionary
# =============================================================================
def create_machine_solutions(chromosome):
    #Create dataframe with the same sequence as the chromosome. The purpose of the frame is to add the operation number immediately.
    frame = pd.DataFrame(np.zeros((len(data), 3)),columns = ['Job','Operation','Machine'])
    list_counter = []
    
    for index in range(0,len(frame)):
        frame.at[index,'Job'] = chromosome[index][0] #Store Job
        frame.at[index,'Machine'] = chromosome[index][1] #Store Machine
        list_counter.append(chromosome[index][0]) #Store all Jobs that were already looked at.
        frame.at[index,'Operation'] = list_counter.count(chromosome[index][0]) #Store Operation, based on how many times the operation is seen already.
        machine_solution_dictionary = {}
        machine_solution = []
    for machines in range(1,M+1):
        machine_solution = []
        for index in range(0,len(frame)):
            if machines == frame.at[index,'Machine']:
                machine_solution.append((int(frame.at[index,'Job']),int(frame.at[index,'Operation'])))
                machine_solution_dictionary[machines] = machine_solution
    return machine_solution_dictionary

# #Test create_machine_solutions(chromosome)
# a = initial_pop(100,File_name)
# chromosome = a[0]
# machine_solution_dictionary = create_machine_solutions(chromosome)
# print(machine_solution_dictionary)

# =============================================================================
# Determine the initial toolset magazine of a machine
# =============================================================================

def initial_magazine(Machine_solution):
    #determine initial tool magazine TM1 and the succeeding operations OS:
    Initial_toolset = []
    Tool_set_load = 0
    OS_range = []
    flag = True
    for i in Machine_solution:
        for index in range(len(data)):
            if i[0] == data.at[index, 'Job'] and i[1] == data.at[index, 'Operation'] and flag == True:
                if Tool_set_load + data.at[index, 'Tool_set_size'] <= C:
                    OS_range.append(data.at[index, 'Tool_set'])
                    if data.at[index, 'Tool_set'] not in Initial_toolset:
                        Initial_toolset.append(data.at[index, 'Tool_set'])
                        Tool_set_load += data.at[index, 'Tool_set_size']  
                else:
                    flag= False
    return Initial_toolset, Tool_set_load, OS_range

# #Test initial_magazine
# Example_3 = [(2, 1), (5, 1), (3, 1), (6, 1), (4, 1), (3, 2)]
# Initial_toolset, Tool_set_load, OS_range = initial_magazine(Example_3)
# print('Initial toolset: ', Initial_toolset)
# print('Tool_set_load: ', Tool_set_load)
# print('OS_range: ', OS_range)
# ===============================

# =============================================================================
# Tool_sets_and_scores: input is the interpreted machine solution from create_machine_solutions(chromosome)
# =============================================================================

def tool_sets_and_scores(Machine_solution, current_operation, current_tool_magazine):
    #Determine succeeding operations OS
    for index, operation in enumerate(Machine_solution):
        if operation == current_operation:
            index_current_operation = index
    OS = Machine_solution[index_current_operation:]
    
    #Determine TSM1: succeeded tool_sets that are still to be installed AFTER the required tool replacement   
    TSM1 = []
    for i in range(1, len(OS)):
        for index in range(len(data)):
            if OS[i][0] == data.at[index, 'Job'] and OS[i][1] == data.at[index, 'Operation']:
                TSM1.append(data.at[index, 'Tool_set'])
                
    #Determine first time toolsets OUS
    OUS = []
    #This dictionary keeps the tool_set that is FIRST required only, and keeps the index in OS as value.
    OUS_dict = {}
    for i in range(1, len(OS)):
        for index in range(len(data)):
            if OS[i][0] == data.at[index, 'Job'] and OS[i][1] == data.at[index, 'Operation'] and data.at[index, 'Tool_set'] not in OUS_dict:
                    OUS.append((OS[i][0] , OS[i][1]))
                    OUS_dict[data.at[index, 'Tool_set']] = i             
    Score = {}
    for i in OUS:
        for index in range(len(data)):
            if i[0] == data.at[index, 'Job'] and i[1] == data.at[index, 'Operation']:
                Score[data.at[index, 'Tool_set']] = len(TSM1) - (OUS_dict.get(data.at[index, 'Tool_set']) - 1)

    for i in current_tool_magazine:
        if i not in TSM1:
            Score[i] = 0
    return Score, OS, index_current_operation
    

# #Test tool_sets_and_scores(Machine_solution, current_operation, current_tool_magazine)
# Example_3 = [(2, 1), (5, 1), (3, 1), (6, 1), (4, 1), (3, 2)]
# current_tool_magazine = [4,2,1]
# Score, OS, index_current_operation = tool_sets_and_scores(Example_3, Example_3[3], current_tool_magazine)
# print(index_current_operation)





# =============================================================================
# ILP: 
# =============================================================================
#input: current_toolset,index_current_operation, Score, Machine_solution
# output: Toolsets to be removed from the current magazine in a list.
 
def ILP(current_toolset,index_current_operation, Score, Machine_solution):
    #Determine Sufficient capacity
    PhiTm = 0 # this is the toolset-load of the current toolset
    PhiTij = int(data.loc[(data["Job"] == Machine_solution[index_current_operation][0]) & (data["Operation"] == Machine_solution[index_current_operation][1]), "Tool_set_size"]) #this is the toolset size of the required toolset
    PhiT = {}
    for i in current_toolset:
        PhiT[i] = data.at[i, 'Tool_set_size']
        
    for i in current_toolset:
        flag = True
        for index in range(len(data)):
            if i == data.at[index, 'Tool_set'] and flag == True:
                PhiTm += data.at[index, 'Tool_set_size']
                flag = False
                
    Sufficient_capacity = PhiTij - (C - PhiTm)
    
    #Determine the size of toolsets maintained with score=0
    Potentially_remove_toolsets = []
    Size_toolsets_0 = 0
    for j in current_toolset:
        for i in Score:
            flag = True
            for index in range(len(data)):
                if i == j == data.at[index, 'Tool_set'] and Score[i] == 0 and flag == True:
                    Size_toolsets_0 += data.at[index, 'Tool_set_size']
                    Potentially_remove_toolsets.append(i)
                    flag = False
                    
    #if there is suffcient capacity without having to remove toolsets, then the list is empty
    if PhiTij <= (C - PhiTm):
        Remove_toolsets = []
    #If size of toolsets in the current toolset with score 0 is larger than sufficient capacity, then remove these toolsets
    elif Size_toolsets_0 >= Sufficient_capacity:
        Remove_toolsets = Potentially_remove_toolsets
    #Else, we use the ILP model.
    else:
        Remove_toolsets = []
        m = gp.Model('ILP')
        m.setParam('OutputFlag', 0)
        x = m.addVars(current_toolset, obj=Score, vtype=GRB.BINARY, name="x")
        m.addConstr(quicksum(x[t]*PhiT[t] for t in current_toolset) >= Sufficient_capacity)
        m.optimize()
        if m.status == GRB.OPTIMAL:
            solution_x = m.getAttr('x', x)
            for t in current_toolset:
                if solution_x[t] > 0:
                    Remove_toolsets.append(t)
    return Remove_toolsets
                     
    


#Test ILP
# Example_3 = [(2, 1), (5, 1), (3, 1), (6, 1), (4, 1), (3, 2)]
# current_toolset = [4,2,1]
# Score, OS, index_current_operation = tool_sets_and_scores(Example_3, (6, 1), [4,2,1])
# ILP_Remove_toolsets = new_ILP(current_toolset,index_current_operation, Score, Example_3)


# Score, OS, index_current_operation = tool_sets_and_scores(b[4], b[4][3], current_tool_magazine)
# print(Score)
# print(Score)
# Remove_toolsets = ILP(current_tool_magazine, index_current_operation, Score, b[4])
# print(Remove_toolsets)
    

# =============================================================================
# Store the results. Input is the solution in the form of a dictionary, e.g.: {1:[(2, 1), (5, 1), (3, 1), (6, 1), (4, 1), (3, 2)]}
# =============================================================================
def result_per_chromosome(machine_solution_dictionary):
    results = data.copy()
    results['chosen machine'] = len(results['Job']) * [0]
    results['Tools equip during process'] =  [ [] for _ in range(len(results['Job'])) ]
    results['start time (A)'] = len(results['Job']) * [0]
    results['end time (E)'] = len(results['Job']) * [0]
    results['set up (Z)'] = len(results['Job']) * [0]
    
    #Set chosen machine
    for Machine_solution in machine_solution_dictionary:
        for i in machine_solution_dictionary[Machine_solution]:
            for index in range(len(results)):
                if i[0] == results.at[index, 'Job'] and i[1] == results.at[index, 'Operation']:
                    results.at[index, 'chosen machine'] = Machine_solution       
    #Set initial magazine
    for Machine_solution in machine_solution_dictionary:
        Initial_toolset, Tool_set_load, OS_range = initial_magazine(machine_solution_dictionary[Machine_solution])
        flag = True
        for i in range(len(machine_solution_dictionary[Machine_solution])):
            for index in range(len(results)):
                if machine_solution_dictionary[Machine_solution][i][0] == results.at[index, 'Job'] and machine_solution_dictionary[Machine_solution][i][1] == results.at[index, 'Operation'] and flag == True:
                    results.at[index,'Tools equip during process'] = Initial_toolset
                    flag = False
                    
    #Set subsequent machine magazines
    for m in machine_solution_dictionary:
        for i,machine_solution in enumerate(machine_solution_dictionary[m]):
            for index in range(len(results)):
                if machine_solution[0] == results.at[index, 'Job'] and machine_solution[1] == results.at[index, 'Operation'] and not results.at[index, 'Tools equip during process']:
                    for j in range(len(data)):
                        if results.at[index, 'Job'] == data.at[j, 'Job'] and results.at[index, 'Operation'] == data.at[j, 'Operation']:
                            for k in range(len(results)):
                                if machine_solution_dictionary[m][i-1][0] == results.at[k, 'Job'] and machine_solution_dictionary[m][i-1][1] == results.at[k, 'Operation']:
                                    if data.at[j, 'Tool_set'] in results.at[k, 'Tools equip during process']:
                                        results.at[index, 'Tools equip during process'] =  results.at[k, 'Tools equip during process']
                                        results.at[index, 'set up (Z)'] = 0
                                    else:
                                        Score, OS, index_current_operation = tool_sets_and_scores(machine_solution_dictionary[m], machine_solution, results.at[k, 'Tools equip during process'])
                                        Remove_toolsets = ILP(results.at[k, 'Tools equip during process'], index_current_operation, Score, machine_solution_dictionary[m])
                                        new_magazine = results.at[k, 'Tools equip during process'].copy()
                                        #this if-statement is just added for debugging. Dont know if it is necessary anymore
                                        if Remove_toolsets != []:
                                            for z in Remove_toolsets:
                                                new_magazine.remove(z)
                                        new_magazine.append(results.at[index, 'Tool_set'])
                                        results.at[index, 'Tools equip during process'] =  new_magazine
                                        results.at[index, 'set up (Z)'] = Set_up_time
    
    #Set start times and end times
    for m in machine_solution_dictionary:
        for i, operation in enumerate(machine_solution_dictionary[m]):
            for index in range(len(results)):
                if operation[0] == results.at[index, 'Job'] and operation[1] == results.at[index, 'Operation']:
                    #Determine start and end times for the first operation on all machines
                    if i == 0:
                        results.at[index, 'start time (A)'] = results.at[index, 'Release_time']
                        results.at[index, 'end time (E)'] = results.at[index, 'start time (A)'] + results.at[index, 'Processing_time']
                    #Determine start and end times for the remaining operations
                    else:
                        #If it concerns the first operation of a job
                        if results.at[index, 'Operation'] == 1:
                            results.at[index, 'start time (A)'] = max(float(results.loc[(results["Job"] == machine_solution_dictionary[m][i-1][0]) & (results["Operation"] == machine_solution_dictionary[m][i-1][1]), "end time (E)"]), float(results.at[index, 'Release_time'])) + float(results.at[index, 'set up (Z)'])
                            results.at[index, 'end time (E)'] = results.at[index, 'start time (A)'] + results.at[index, 'Processing_time']
                        #for the other operations we include the end time of the previous operation. Operation 1 should always be completed before operation 2, even it is performed on another machine.
                        else:
                            results.at[index, 'start time (A)'] = max(float(results.loc[(results["Job"] == results.at[index, "Job"]) & (results["Operation"] == results.at[index, 'Operation']-1), "end time (E)"]), float(results.loc[(results["Job"] == machine_solution_dictionary[m][i-1][0]) & (results["Operation"] == machine_solution_dictionary[m][i-1][1]), "end time (E)"]), float(results.at[index, 'Release_time'])) + float(results.at[index, 'set up (Z)'])
                            results.at[index, 'end time (E)'] = results.at[index, 'start time (A)'] + results.at[index, 'Processing_time']
    #Sort results table based on start times.                       
    results = results.sort_values(by=['start time (A)'])
    
    #Set total endtime (for possible improvement), total tardiness, total setuptime
    Total_endtime = results['end time (E)'].max()
    Total_tardiness = 0
    Total_setup_time = 0
    for index in range(len(results)):
        if float(results.at[index, 'end time (E)']) > float(results.at[index, 'Due_date']):
            Total_tardiness += float(results.at[index, 'end time (E)']) - float(results.at[index, 'Due_date'])
        Total_setup_time += results.at[index, 'set up (Z)']
            
    #Fitness evaluation
    Fitness = Wd * Total_tardiness + Ws * Total_setup_time
    return Fitness

# # test result_per_chromosome(Machine_solution)
# a = initial_pop(20,File_name)
# Example_chromosome = a[7]

# Example_solution = create_machine_solutions(Example_chromosome)
# Fitness = result_per_chromosome(Example_solution)
# print('Fitness: ', Fitness)

# =============================================================================
# =============================================================================
# # Mutation with probability of P_mutation. The mutation swaps a random gene, with another random gene.
# =============================================================================
# Input: a mutation candidate in the form of a chromosome
# Output: the mutated chromosome
# =============================================================================


def swap_mutation(chromosome):
    mutated_chromosome = chromosome.copy()
    chromosome_copy = chromosome.copy()
    for gene_index in range(len(chromosome_copy)):
        if random.random() <= Ps:
            #choose a swap target which is an index in range(0, O) and is not equal to the index of the selected gene
            swap_target = choice([i for i in range(O) if i != gene_index])
            mutated_chromosome[gene_index] = (chromosome_copy[swap_target][0], chromosome_copy[gene_index][1])
            mutated_chromosome[swap_target] = (chromosome_copy[gene_index][0], chromosome_copy[swap_target][1])
            chromosome_copy = mutated_chromosome.copy()
    return mutated_chromosome



# #Test swap_mutation
# a = initial_pop(100,File_name)
# chromosome1 = a[0]
# mutated_chromosome = swap_mutation(chromosome1)


# =============================================================================
# =============================================================================
# # Uniform mutation: input is a chromosome, output is the mutated chromosome
# =============================================================================
# =============================================================================
def uniform_mutation(chromosome):
    mutated_chromosome = chromosome.copy()
    for gene_index in range(len(chromosome)):
        if random.random() <= Pu:
            mutated_chromosome[gene_index] = (chromosome[gene_index][0], random.randint(1,M))
    return mutated_chromosome
            
    
# #Test uniform_mutation:
# a = initial_pop(100,File_name)
# chromosome1 = a[0]
# Mutated_chromosome = uniform_mutation(chromosome1)

def elitism(population):
    Parents_with_fitness = {}
    count = 0
    #Select the SE best chromosomes of a population (SE is defined in parameter section)
    for chromosome in population:
          machine_solution_dictionary = create_machine_solutions(chromosome)
          Fitness = result_per_chromosome(machine_solution_dictionary)
          Parents_with_fitness[tuple(chromosome)] = Fitness          
          count += 1/len(population)
          print('Elitism function progress: ', int(count * 100), '%')
    Sorted_parents_with_fitness = dict(sorted(Parents_with_fitness.items(), key=lambda item: item[1]))
    Best_parents = []
    for i in range(SE):
        Best_parents.append(list(list(Sorted_parents_with_fitness.keys())[i]))
        
    # # with the 2 lines below you can check if the fitness values of the chromosomes are indeed the lowest ones.
    # # Tuple is necessary, because the dictionary can only be accessed by tuples and not lists!
    # for i in Best_parents:
    #     print(Sorted_parents_with_fitness[tuple(i)])
    
    #Next step: Replace randomly selected offspring with the best parents defined above
    return Best_parents

# ====================================================================================================================================

def chromosome_generator():
    #Change format of input (Input is list with EDD sequence of Jobs)
    table = data.copy()
    table = table.sort_values('Due_date',ignore_index=True)
    Job_lst = []
    for index in range(0,len(table)):
        Job_lst.append(table.at[index,'Job'])
      
    #Job list will be the output of the V^I chromosome part:
    job_list_output = []
    
    # Repeat till chromosome has required length.
    while len(job_list_output) < len(Job_lst):
        how_often = len(job_list_output)
        
        #First create list of all jobs that are within the range (already used is not yet considered)
        option_list = []
        for i in Job_lst[0:int(min(how_often+n_parameter*len(Job_lst),len(Job_lst)))]:
            option_list.append(i)
       
        #Now remove all already used job values.
        for i in job_list_output:
            option_list.remove(i)
        
        #Now a random job can be chosen from the remaining options
        chosen_job = random.choice(option_list)
        job_list_output.append(chosen_job)
    
    #Make chromosomes
    Chromosome = []
    for i in job_list_output:
        Chromosome.append((i, random.choice(machines)))
    return Chromosome    

print(chromosome_generator())



# ====================================================================================================================================

def initial_pop(Population_Size,CSV_file_name):
    population = []
    
    # First we will add the chromosome from the PH
    PH_chrom = PH_Chromosome_generator(CSV_file_name)
    population.append(PH_chrom)
    
    # Secondly, we will add the remaining Chromosomes till population is filled.
    while len(population) < Population_Size:
        Chrom_generated = chromosome_generator()
        if Chrom_generated not in population:
            population.append(Chrom_generated)
    return population

#Testing Initial_pop
#print('Initial population')
#a = initial_pop(100,File_name)
#print('length population =', len(a))
#print('chrom 1:   ', a[0])
#print('chrom 2:   ', a[1])
#print('chrom 3:   ', a[2])
#print('....')
#print('_____________________________', '\n')
    
# ====================================================================================================================================

def tournament(input_poplulation,Fitness):     # Initial_pop = lst of chromosomes, Fitness = list of CORRESPONDING Fitness values, values located at corresponding indexes in second list.
    #Determine how many competitor chromosomes there are (S_t)
    S_t = Gamma1 * len(input_poplulation)
    ###TEST###print('S_t',S_t)
    
    # Retrieve S_t unique index values so that by those indexes the chromosomes can be chosen.
    test = 0
    tournament_indexes_lst = []
    while test < S_t:
        number = int(random.uniform(0,len(input_poplulation)))
        if number not in tournament_indexes_lst:
            tournament_indexes_lst.append(number)
            test = test + 1
    
    # Based on the random chosen indexes, the corresponding chromosomes and fitness values are retrieved and put in a list.
    output_tournament_cr = []
    output_tournament_fn = []
    for index in tournament_indexes_lst:
        output_tournament_cr.append(input_poplulation[index])
        output_tournament_fn.append(Fitness[index])
    ###TEST###print(tournament_indexes_lst)
    ###TEST###print(len(output_tournament_cr))
    
    
    #The retrieved values are put in a dataframe, which can be sorted (based on the fitness). The lowest fitness value chromosome is the winner.
    tf =  pd.DataFrame(index=range(0,len(output_tournament_cr)), columns=['Chromosoom'])
    tf['Chromosoom'] = output_tournament_cr
    tf['Fitness'] = output_tournament_fn
    tf = tf.sort_values('Fitness',ignore_index=True)
    parent_selected = tf.at[0,'Chromosoom']
    return parent_selected

# ====================================================================================================================================

def POX_Vm_assignment(Vi_part_of_chromosome):   # This function is used in the POX crossover for V^M (it is algorithm 2 in the paper)
    # ---------------------------------------------------------
    #Create table based on Vi_part_of_chromosome, which determines Operation number and has space to store the resulting Machine name. Look for Process time and toolset + size
    Chrom_one = pd.DataFrame(Vi_part_of_chromosome, columns = ['Job'])
    Chrom_one['Operation'] = [0] * len(Chrom_one['Job'])
    
    #Find out what operation number is required.
    count_list_one = []
    for index in range(0,len(Chrom_one)):
        count_list_one.append(Vi_part_of_chromosome[index])
        Chrom_one.at[index,'Operation'] = count_list_one.count(Vi_part_of_chromosome[index])
    
    #Create empty machine assignment to store results in.
    Chrom_one['Machine'] = [0] * len(Chrom_one['Job'])
    
    #Retrieve tool name, tool_size and processing time from dataset.
    for index in range(0,len(Chrom_one)):
        Chrom_one.at[index,'Tool'] = int(data.loc[(data['Job'] == Chrom_one.at[index,'Job']) & (data['Operation'] == Chrom_one.at[index,'Operation']),"Tool_set"])
        Chrom_one.at[index,'Tool_size'] = int(data.loc[(data['Job'] == Chrom_one.at[index,'Job']) & (data['Operation'] == Chrom_one.at[index,'Operation']),"Tool_set_size"])
        Chrom_one.at[index,'Process_time'] = float(data.loc[(data['Job'] == Chrom_one.at[index,'Job']) & (data['Operation'] == Chrom_one.at[index,'Operation']),"Processing_time"])
    
    # --------------------------
    #List of machines (naming)
    MachineList = list(range(1,M+1)) 
    # -------------------------- 
    #M_overview_one - stores tools selected / corresponding tool size / total process time of assigned jobs.
    M_overview_one = pd.DataFrame(np.zeros((M,1)))
    M_overview_one.columns = ['Sel_tool_size']
    M_overview_one.index = MachineList
    M_overview_one['Sel_tools'] = [ [] for _ in range(M) ]
    M_overview_one['Tot_process'] = 0 * len(M_overview_one)
    # --------------------------
    for index in range(0,len(Chrom_one)):
        
        #Machines with tool selected are:
        for machine in MachineList:
            if Chrom_one.at[index,'Tool'] in M_overview_one.at[machine,'Sel_tools']:
                Chrom_one.at[index,'Machine'] = machine # Machine chosen
                M_overview_one.at[machine,'Tot_process'] = M_overview_one.at[machine,'Tot_process'] + Chrom_one.at[index,'Process_time'] #Process time is added by the total
        
        if Chrom_one.at[index,'Machine'] == 0:  #Which means it is not assigned yet!
            
            #Make a list of machines with enough space.
            lst_mach_enough_space = []
            for machine in MachineList:
                req_space = Chrom_one.at[index,'Tool_size']
                if C - M_overview_one.at[machine,'Sel_tool_size'] >= req_space:         #if C (capacity) - used space >= required space, remember machine.
                    lst_mach_enough_space.append(machine)
            
            if lst_mach_enough_space != []: #If machine with space is found!
                M_proc = 0 #Initial value that is not a machine at all.
                M_proc_total = float('inf') #At first an unrealisticly high number is choosen. So that the first option is always earlier.
                for mach in lst_mach_enough_space:
                    if M_overview_one['Tot_process'][mach] < M_proc_total:
                        M_proc_total = M_overview_one['Tot_process'][mach]
                        M_proc = mach
                
                #Given that M_proc is the chosen machine, append + add toolsize + process time:
                M_overview_one.at[M_proc,'Tot_process'] = M_overview_one.at[M_proc,'Tot_process'] + Chrom_one.at[index,'Process_time']
                M_overview_one.at[M_proc,'Sel_tools'].append(int(Chrom_one.at[index,'Tool']))
                M_overview_one.at[M_proc,'Sel_tool_size'] = M_overview_one.at[M_proc,'Sel_tool_size'] + Chrom_one.at[index,'Tool_size']
                Chrom_one.at[index,'Machine'] = M_proc
            
            else: # If no machine has the required space! We will only review the shortest Total processing time.
                M_proc = 0 #Initial value that is not a machine at all.
                M_proc_total = float('inf') #At first an unrealisticly high number is choosen. So that the first option is always earlier.
                for mach in MachineList:
                    if M_overview_one['Tot_process'][mach] < M_proc_total:
                        M_proc_total = M_overview_one['Tot_process'][mach]
                        M_proc = mach
                
                #Given that the fastest machine is chosen (we will assign it and add the process time. Note that toolset+tool_size wont chance as there is no space)
                M_overview_one.at[M_proc,'Tot_process'] = M_overview_one.at[M_proc,'Tot_process'] + Chrom_one.at[index,'Process_time']
                Chrom_one.at[index,'Machine'] = M_proc
    
    #Create output list of (Job,Machine)
    output_chrom = []
    for index in range(0,len(Chrom_one)):
        output_chrom.append((Chrom_one.at[index,'Job'],Chrom_one.at[index,'Machine']))  
    ###TEST###print(output_chrom)
    return output_chrom


# ====================================================================================================================================


def crossover_POX(parent_one,parent_two):
    ############################################################################
    # STEP 0: Converting the input of function to prefered format (dataframe)
    ############################################################################
    # Parent one is converted into a dataframe.
    P_one = pd.DataFrame(np.zeros((len(parent_one),2)), columns=['Job','Machine'])
    for index in range(0,len(P_one['Job'])):
        P_one.at[index,'Job'] = parent_one[index][0]
        P_one.at[index,'Machine'] = parent_one[index][1]
    
    # Parent two is converted into a dataframe.
    P_two = pd.DataFrame(np.zeros((len(parent_two),2)), columns=['Job','Machine'])
    for index in range(0,len(P_two['Job'])):
        P_two.at[index,'Job'] = parent_two[index][0]
        P_two.at[index,'Machine'] = parent_two[index][1]
        
    # Determine Operation value in P_one:
    tel_lijst_one = []
    for index in range(0,len(P_one)):
        tel_lijst_one.append(P_one.at[index,'Job'])
        P_one.at[index,'Operation'] = tel_lijst_one.count(P_one.at[index,'Job'])
    
    # Determine Operation value in P_two
    tel_lijst_two = []
    for index in range(0,len(P_two)):
        tel_lijst_two.append(P_two.at[index,'Job'])
        P_two.at[index,'Operation'] = tel_lijst_two.count(P_two.at[index,'Job'])
    
    ############################################################################
    # STEP 1: For Parent one & two, apply transformed scheme
    ############################################################################
    # For p_one, the transformation scheme is literally (1,2,3,4,5,6,.....,len(P_one))    
    P_one['transform'] = list(set(range(1,len(P_one)+1)))
    
    # For p_two, these values must be looked up based on Job & Operation values.
    for index in range(0,len(P_two)):
        P_two.at[index,'transform'] = int(P_one.loc[(P_one['Job'] == P_two['Job'][index]) & (P_one['Operation'] == P_two['Operation'][index]),"transform"])
    
    ############################################################################
    # STEP 2: For Parent one & two, the part between two cutpoints gets swapped
    ############################################################################
    
    # We need two cutpoints. They are sorted afterwards so that Cutpoint 1 is the lower number.
    Cutpoint_one = int(random.uniform(0,len(P_one['Job'])))
    getal=0
    while getal < 1:
        Cutpoint_two= int(random.uniform(0,len(P_one['Job'])))
        if Cutpoint_one != Cutpoint_two:
            getal = getal + 1
    
    if Cutpoint_one > Cutpoint_two:
        remember = Cutpoint_one
        Cutpoint_one = Cutpoint_two
        Cutpoint_two = remember
        
    ###TEST###print('Cut1', Cutpoint_one)
    ###TEST###print('Cut2', Cutpoint_two)
    
    # -_-_-_-_-_-_-_-_-_-_-_-_- Tussendoor in stap 2: INFORMATIE OPSLAAN VOOR STAP 3 & 4 -_-_-_-_-_-_-_-_-_-_-_-_- 
    # The swaps between the cutpoints need to be remembered. (USED FOR STEP 3, PLAATJE LINKERKANT)
    T1 = pd.DataFrame(np.zeros((len(range(Cutpoint_one,Cutpoint_two+1)),2)), columns=['transform_P1','transform_P2'])
    for index in range(0,len(T1)):
        T1.at[index,'transform_P1'] = P_one.at[index+Cutpoint_one,'transform']
        T1.at[index,'transform_P2'] = P_two.at[index+Cutpoint_one,'transform']
    
    # At the end we need to know how to translate the encripted values back. A current copy of P_one can be used for this now.
    Translate_back = P_one.copy()
    Translate_back = Translate_back.drop('Machine',1)
    #  -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    
    #Swap the transformed scheme between the cutpoints (inclusive of the cutpoint numbers index) between the two parents.
    for index in range(Cutpoint_one,Cutpoint_two+1): #Both numbers inclusive
        P_one.at[index,'transform'] = P_two.at[index,'transform']
        P_two.at[index,'transform'] = Translate_back.at[index,'transform'] # This value already got the P_one.copy() values stored.
    
    #Drop all columns, except for 'transformed scheme' to make sure no mistakes are made, as transformed values will get swapped again in step 3.
    PO_een = P_one.copy().drop(['Job','Operation','Machine'],1)
    PO_twee = P_two.copy().drop(['Job','Operation','Machine'],1)
    
    ############################################################################
    #STEP 3 (make sure there are no duplicute transformed values are left. This is 100% surely fixed by Check_x_times = len(T1), however for speed reasons = round(len(T1)*0.5)) seems to work fine. IF ERROR OCCURS FIX THIS BY WAY EXPLAINED HERE!!!!!
    ############################################################################
    ###TEST###count = 0
    Check_x_times = 0
    while Check_x_times < len(T1): # INCREASE TO FIX ERROR, AT THE COST OF RUN TIME!
        ####print('len',len(T1))
        if Cutpoint_one != 0:
            for index in range(0,Cutpoint_one):
                if PO_een.at[index,'transform'] in T1['transform_P2'].values:
                    ###TEST###print(index)
                    ###TEST###print(PO_een.at[index,'transform'], '--->',int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"]) )
                    PO_een.at[index,'transform'] = int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"])
                    ###TEST###count += 1
        
        if Cutpoint_two != len(PO_een)-1:
            for index in range(Cutpoint_two+1,len(PO_een)):
                if PO_een.at[index,'transform'] in T1['transform_P2'].values:
                    PO_een.at[index,'transform'] = int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"])
                    ###TEST###count += 1
    
        #Repeat the same steps for P_twee
        if Cutpoint_one != 0:
            for index in range(0,Cutpoint_one):
                if PO_twee.at[index,'transform'] in T1['transform_P1'].values:
                    PO_twee.at[index,'transform'] = int(T1.loc[(T1["transform_P1"].values ==PO_twee.at[index,'transform']), "transform_P2"])
        
        if Cutpoint_two != len(PO_twee)-1:
            for index in range(Cutpoint_two+1,len(PO_twee)):
                if PO_twee.at[index,'transform'] in T1['transform_P1'].values:
                    PO_twee.at[index,'transform'] = int(T1.loc[(T1["transform_P1"].values ==PO_twee.at[index,'transform']), "transform_P2"])
        
        Check_x_times += 1
        
        #However you can stop earlier when there are no duplicute values anymore in both the PO_een and PO_twee.
        dup_een = len(PO_een.groupby('transform').filter(lambda x: len(x) > 1).drop_duplicates(subset='transform'))
        dup_twee = len(PO_twee.groupby('transform').filter(lambda x: len(x) > 1).drop_duplicates(subset='transform'))
        ###TEST###print(dup_een, dup_twee)
        if dup_een == 0 and dup_twee == 0:
            Check_x_times = len(T1)         #So, if there are no duplicutes any more, you can stop the while loop.
    
    ############################################################################
    # STEP 4: Translating back, based on the Translate_back table created earlier.
    ############################################################################
    
    #PO_een, retrieve job value.
    for index in range(0,len(PO_een)):
        PO_een.at[index,'Job'] = int(Translate_back.loc[(Translate_back['transform'].values == PO_een.at[index,'transform']), "Job"])
    
    #PO_twee, retrieve job value 
    for index in range(0,len(PO_twee)):
        PO_twee.at[index,'Job'] = int(Translate_back.loc[(Translate_back['transform'].values == PO_twee.at[index,'transform']), "Job"])
    
    ############################################################################
    # STEP 5: Outside Cutpoints - Eearliest due date sequencing.
    ############################################################################   
    
    #------------------
    #STEP EDD - Offspring 1 - dataframe:
    EDD_offspring_one = pd.DataFrame(np.zeros((len(PO_een),3)), columns = ['Job','Operation','Due_date'])
    lst_count = []
    for i in range(0,len(PO_een)):
        EDD_offspring_one.at[i,'Job'] = PO_een.at[i,'Job']
        lst_count.append(PO_een.at[i,'Job'])
        EDD_offspring_one.at[i,'Operation'] = lst_count.count(PO_een.at[i,'Job'])
        EDD_offspring_one.at[i,'Due_date'] = float(data.loc[(data['Job'].values == int(EDD_offspring_one.at[i,'Job'])) & (data['Operation'] == int(EDD_offspring_one.at[i,'Operation']) ),'Due_date'])
    
    # Create table of sorted on EDD for offspring 1 in a new table (remember) while having dropped the values between cutpoints.
    EDD_offspring_remember = EDD_offspring_one.copy()
    EDD_offspring_remember = EDD_offspring_remember.drop(range(Cutpoint_one,Cutpoint_two+1))
    EDD_offspring_remember_sorted = EDD_offspring_remember.sort_values('Due_date',ignore_index=True)
    
    # Combine both tables again to get the result                     
    indexes_outside_cutpoints = list(range(0,Cutpoint_one)) + list(range(Cutpoint_two+1,len(PO_een)))
    index_sorted = 0
    for index in indexes_outside_cutpoints:
        EDD_offspring_one.at[index,'Job'] = EDD_offspring_remember_sorted.at[index_sorted,'Job']
        EDD_offspring_one.at[index,'Operation'] = EDD_offspring_remember_sorted.at[index_sorted,'Operation']
        EDD_offspring_one.at[index,'Due_date'] = EDD_offspring_remember_sorted.at[index_sorted,'Due_date']
        index_sorted = index_sorted + 1
     #------------------
    #STEP EDD - Offspring 2 - dataframe:
    EDD_offspring_two = pd.DataFrame(np.zeros((len(PO_twee),3)), columns = ['Job','Operation','Due_date'])
    lst_count = []
    for i in range(0,len(PO_twee)):
        EDD_offspring_two.at[i,'Job'] = PO_twee.at[i,'Job']
        lst_count.append(PO_twee.at[i,'Job'])
        EDD_offspring_two.at[i,'Operation'] = lst_count.count(PO_twee.at[i,'Job'])
        EDD_offspring_two.at[i,'Due_date'] = float(data.loc[(data['Job'].values == int(EDD_offspring_two.at[i,'Job'])) & (data['Operation'] == int(EDD_offspring_two.at[i,'Operation']) ),'Due_date'])
    
    # Create table of sorted on EDD for offspring 2 in a new table (remember) while having dropped the values between cutpoints.
    EDD_offspring_remember = EDD_offspring_two.copy()
    EDD_offspring_remember = EDD_offspring_remember.drop(range(Cutpoint_one,Cutpoint_two+1))
    EDD_offspring_remember_sorted = EDD_offspring_remember.sort_values('Due_date',ignore_index=True)
    
    # Combine both tables again to get the result                     
    indexes_outside_cutpoints = list(range(0,Cutpoint_one)) + list(range(Cutpoint_two+1,len(PO_twee)))
    index_sorted = 0
    for index in indexes_outside_cutpoints:
        EDD_offspring_two.at[index,'Job'] = EDD_offspring_remember_sorted.at[index_sorted,'Job']
        EDD_offspring_two.at[index,'Operation'] = EDD_offspring_remember_sorted.at[index_sorted,'Operation']
        EDD_offspring_two.at[index,'Due_date'] = EDD_offspring_remember_sorted.at[index_sorted,'Due_date']
        index_sorted = index_sorted + 1
    
    ############################################################################
    # STEP V^M: Add to the V^I the machine based on algorithm 2 (paper Dang) - details in function: "POX_Vm_assignment()"
    ############################################################################
    #Create list of the jobs for P_een, as input for the POX_Vm_assignment() function.
    Chrom_Vi_een = []
    for index in range(0,len(EDD_offspring_one)):
        Chrom_Vi_een.append(int(EDD_offspring_one.at[index,'Job']))
    
    #Create list of the jobs for P_twee, as input for the POX_Vm_assignment() function.
    Chrom_Vi_twee = []  
    for index in range(0,len(EDD_offspring_two)):
        Chrom_Vi_twee.append(int(EDD_offspring_two.at[index,'Job']))    
    
    #Use the POX_Vm_assignment() function:
    OFFSPRING_ONE = POX_Vm_assignment(Chrom_Vi_een)
    OFFSPRING_TWO = POX_Vm_assignment(Chrom_Vi_twee)
    return OFFSPRING_ONE, OFFSPRING_TWO


#Testing function crossover_POX()
#par_een = chromosome_generator()    #INPUT --> Parent 1
#par_twee = chromosome_generator()   #INPUT --> Parent 2
#Output_offspring_een, Output_offspring_twee = crossover_POX(par_een,par_twee)
#print(Output_offspring_een)         #OUTPUT--> Offspring 1
#print(Output_offspring_twee)        #OUTPUT--> Offspring 2

# ====================================================================================================================================   

def crossover_CX(parent_one,parent_two):
    ############################################################################
    # STEP 0: Converting the input of function to prefered format (dataframe)
    ############################################################################
    # Parent one is converted into a dataframe.
    P_one = pd.DataFrame(np.zeros((len(parent_one),2)), columns=['Job','Machine'])
    for index in range(0,len(P_one['Job'])):
        P_one.at[index,'Job'] = parent_one[index][0]
        P_one.at[index,'Machine'] = parent_one[index][1]
    
    # Parent two is converted into a dataframe.
    P_two = pd.DataFrame(np.zeros((len(parent_two),2)), columns=['Job','Machine'])
    for index in range(0,len(P_two['Job'])):
        P_two.at[index,'Job'] = parent_two[index][0]
        P_two.at[index,'Machine'] = parent_two[index][1]
        
    # Determine Operation value in P_one:
    tel_lijst_one = []
    for index in range(0,len(P_one)):
        tel_lijst_one.append(P_one.at[index,'Job'])
        P_one.at[index,'Operation'] = tel_lijst_one.count(P_one.at[index,'Job'])
    
    # Determine Operation value in P_two
    tel_lijst_two = []
    for index in range(0,len(P_two)):
        tel_lijst_two.append(P_two.at[index,'Job'])
        P_two.at[index,'Operation'] = tel_lijst_two.count(P_two.at[index,'Job'])
    
    ############################################################################
    # STEP 1: For Parent one & two, apply transformed scheme
    ############################################################################
    # For p_one, the transformation scheme is literally (1,2,3,4,5,6,.....,len(P_one))    
    P_one['transform'] = list(set(range(1,len(P_one)+1)))
    
    # For p_two, these values must be looked up based on Job & Operation values.
    for index in range(0,len(P_two)):
        P_two.at[index,'transform'] = int(P_one.loc[(P_one['Job'] == P_two['Job'][index]) & (P_one['Operation'] == P_two['Operation'][index]),"transform"])
    
    ############################################################################
    # STEP 2: For Parent one & two, the part between two cutpoints gets swapped
    ############################################################################
    
    # We need two cutpoints. They are sorted afterwards so that Cutpoint 1 is the lower number.
    Cutpoint_one = int(random.uniform(0,len(P_one['Job'])))
    getal=0
    while getal < 1:
        Cutpoint_two= int(random.uniform(0,len(P_one['Job'])))
        if Cutpoint_one != Cutpoint_two:
            getal = getal + 1
    
    if Cutpoint_one > Cutpoint_two:
        remember = Cutpoint_one
        Cutpoint_one = Cutpoint_two
        Cutpoint_two = remember
        
    ###TEST###print('Cut1', Cutpoint_one)
    ###TEST###print('Cut2', Cutpoint_two)
    
    # -_-_-_-_-_-_-_-_-_-_-_-_- Tussendoor in stap 2: INFORMATIE OPSLAAN VOOR STAP 3 & 4 -_-_-_-_-_-_-_-_-_-_-_-_- 
    # The swaps between the cutpoints need to be remembered. (USED FOR STEP 3, PLAATJE LINKERKANT)
    T1 = pd.DataFrame(np.zeros((len(range(Cutpoint_one,Cutpoint_two+1)),2)), columns=['transform_P1','transform_P2'])
    for index in range(0,len(T1)):
        T1.at[index,'transform_P1'] = P_one.at[index+Cutpoint_one,'transform']
        T1.at[index,'transform_P2'] = P_two.at[index+Cutpoint_one,'transform']
    
    # At the end we need to know how to translate the encripted values back. A current copy of P_one can be used for this now.
    Translate_back = P_one.copy()
    Translate_back = Translate_back.drop('Machine',1)
    #  -_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-_-
    
    #Swap the transformed scheme between the cutpoints (inclusive of the cutpoint numbers index) between the two parents.
    for index in range(Cutpoint_one,Cutpoint_two+1): #Both numbers inclusive
        P_one.at[index,'transform'] = P_two.at[index,'transform']
        P_two.at[index,'transform'] = Translate_back.at[index,'transform'] # This value already got the P_one.copy() values stored.
    
    #Drop all columns, except for 'transformed scheme' to make sure no mistakes are made, as transformed values will get swapped again in step 3.
    PO_een = P_one.copy().drop(['Job','Operation','Machine'],1)
    PO_twee = P_two.copy().drop(['Job','Operation','Machine'],1)
    
    ############################################################################
    #STEP 3 (make sure there are no duplicute transformed values are left. This is 100% surely fixed by Check_x_times = len(T1), however for speed reasons = round(len(T1)*0.5)) seems to work fine. IF ERROR OCCURS FIX THIS BY WAY EXPLAINED HERE!!!!!
    ############################################################################
    ###TEST###count = 0
    Check_x_times = 0
    while Check_x_times < len(T1): # INCREASE TO FIX ERROR, AT THE COST OF RUN TIME!
        if Cutpoint_one != 0:
            for index in range(0,Cutpoint_one):
                if PO_een.at[index,'transform'] in T1['transform_P2'].values:
                    ###TEST###print(index)
                    ###TEST###print(PO_een.at[index,'transform'], '--->',int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"]) )
                    PO_een.at[index,'transform'] = int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"])
                    ###TEST###count += 1
        
        if Cutpoint_two != len(PO_een)-1:
            for index in range(Cutpoint_two+1,len(PO_een)):
                if PO_een.at[index,'transform'] in T1['transform_P2'].values:
                    PO_een.at[index,'transform'] = int(T1.loc[(T1["transform_P2"].values ==PO_een.at[index,'transform']), "transform_P1"])
                    ###TEST###count += 1
    
        #Repeat the same steps for P_twee
        if Cutpoint_one != 0:
            for index in range(0,Cutpoint_one):
                if PO_twee.at[index,'transform'] in T1['transform_P1'].values:
                    PO_twee.at[index,'transform'] = int(T1.loc[(T1["transform_P1"].values ==PO_twee.at[index,'transform']), "transform_P2"])
        
        if Cutpoint_two != len(PO_twee)-1:
            for index in range(Cutpoint_two+1,len(PO_twee)):
                if PO_twee.at[index,'transform'] in T1['transform_P1'].values:
                    PO_twee.at[index,'transform'] = int(T1.loc[(T1["transform_P1"].values ==PO_twee.at[index,'transform']), "transform_P2"])
        
        Check_x_times += 1
        
        #However you can stop earlier when there are no duplicute values anymore in both the PO_een and PO_twee.
        dup_een = len(PO_een.groupby('transform').filter(lambda x: len(x) > 1).drop_duplicates(subset='transform'))
        dup_twee = len(PO_twee.groupby('transform').filter(lambda x: len(x) > 1).drop_duplicates(subset='transform'))
        ###TEST###print(dup_een, dup_twee)
        if dup_een == 0 and dup_twee == 0:
            Check_x_times = len(T1)         #So, if there are no duplicutes any more, you can stop the while loop.
    
    ############################################################################
    # STEP 4: Translating back, based on the Translate_back table created earlier.
    ############################################################################
    
    #PO_een, retrieve job value.
    for index in range(0,len(PO_een)):
        PO_een.at[index,'Job'] = int(Translate_back.loc[(Translate_back['transform'].values == PO_een.at[index,'transform']), "Job"])
    
    #PO_twee, retrieve job value 
    for index in range(0,len(PO_twee)):
        PO_twee.at[index,'Job'] = int(Translate_back.loc[(Translate_back['transform'].values == PO_twee.at[index,'transform']), "Job"])
        
    #Drop transformation scheme:
    PO_een = PO_een.drop('transform',1)
    PO_twee= PO_twee.drop('transform',1)
     
    ############################################################################
    # STEP V^M: Crossover between two (new) cutpoints selected cutpoints.
    ############################################################################
    
    #Add original machine order to corresponding indexes
    for index in range(0,len(PO_een['Job'])):
        PO_een.at[index,'Machine'] = parent_one[index][1]
        PO_twee.at[index,'Machine'] = parent_two[index][1]
    
    # Two random cutpoints are chosen. The second one cannot be equal to the first. They are sorted afterwards so that Cutpoint 1 is the lower number.
    Cutpoint_one= int(random.uniform(0,len(PO_een['Machine'])))
    test=0
    while test < 1:
        Cutpoint_two= int(random.uniform(0,len(PO_een['Machine'])))
        if Cutpoint_one != Cutpoint_two:
            test = test + 1
    
    if Cutpoint_one > Cutpoint_two:
        remember = Cutpoint_one
        Cutpoint_one = Cutpoint_two
        Cutpoint_two = remember
    
    # A dataframe is created in which the values can be stored temperarily. So that they can be swapped between P_one and P_two
    P_remember = pd.DataFrame(np.zeros((len(PO_twee),1)), columns=['Machine'])
    
    for index in range(Cutpoint_one,Cutpoint_two+1):
        P_remember.at[index,'Machine'] = PO_een.at[index,'Machine']
        PO_een.at[index,'Machine'] = PO_twee.at[index,'Machine']
        PO_twee.at[index,'Machine'] = P_remember.at[index,'Machine']
    
    # Convert results from 'PO_een & PO_twee' to list with tuples (Job, Machine)
    OFFSPRING_EEN_OUT = []
    OFFSPRING_TWEE_OUT = []
    for index in range(0,len(PO_een)):
        OFFSPRING_EEN_OUT.append( (int(PO_een.at[index,'Job']) , int(PO_een.at[index,'Machine'])) )
        OFFSPRING_TWEE_OUT.append( (int(PO_twee.at[index,'Job']) , int(PO_twee.at[index,'Machine'])) )    
    return OFFSPRING_EEN_OUT , OFFSPRING_TWEE_OUT

#Testing crossover_CX() function
#par_een = chromosome_generator()    #INPUT --> Parent 1
#par_twee = chromosome_generator()   #INPUT --> Parent 2
#Output_offspring_een, Output_offspring_twee = crossover_CX(par_een,par_twee)
#print(Output_offspring_een)         #OUTPUT--> Offspring 1
#print(Output_offspring_twee)        #OUTPUT--> Offspring 2





############################################################################################################################
############################################################################################################################
############################################################################################################################

#Start timer:
Start_MH = timeit.default_timer()       #=start time (0)

#Line 1:
k = 1
best = False
w = float('inf')

#Line 2:
Initialize_Pk = initial_pop(NP,File_name)

Store_F_Chrom = pd.DataFrame(np.zeros((NP, 1)),columns = ['Fitness'])
Store_F_Chrom['Chromosome'] = [ [] for _ in range(NP) ]

#Line 3-4:
for index in range(0,NP):
    chromosome = Initialize_Pk[index]
    Store_F_Chrom.at[index,'Chromosome'] = chromosome  
    machine_solution_dictionary = create_machine_solutions(chromosome)   
    Fitness = result_per_chromosome(machine_solution_dictionary)
    Store_F_Chrom.at[index,'Fitness'] = Fitness
    print('Iteration',0,'   -->',' Evaluate for chromosome',index)

#Line 5:   
Store_F_Chrom = Store_F_Chrom.sort_values('Fitness',ignore_index=True)
f_best_Fitness = Store_F_Chrom.at[0,'Fitness']
f_best_Chromosome = Store_F_Chrom.at[0,'Chromosome']
print('Iteration',0,'   -->',' Best Fitness value',f_best_Fitness)

#Line 6:
q = 1       #Initial value

while (timeit.default_timer()-Start_MH) < Max_time and q < Gc:             #Timer and no improvements

    # convert data:
    list_chrom = []
    list_Fitness = []
    for index in range(0,NP):
        list_chrom.append(Store_F_Chrom.at[index,'Chromosome'])
        list_Fitness.append(Store_F_Chrom.at[index,'Fitness'])
    
    Ck = []
    while len(Ck) < NP:
    
        #Select two unique parents:
        parent_one = tournament(list_chrom,list_Fitness)
        ###print('Parent_one',parent_one)
        getal = 0
        while getal == 0:
            parent_two = tournament(list_chrom,list_Fitness)
            ###print('Parent_two',parent_two)
            if parent_one != parent_two:
                getal = 1
    
    #Line 7 + 8:
        if best == True or q <= Beta:       
           Offspring_one, Offspring_two = crossover_POX(parent_one,parent_two)
           Ck.append(Offspring_one)
           Ck.append(Offspring_two)
    
    #Line 9-10:
        else:
            Offspring_one, Offspring_two = crossover_CX(parent_one,parent_two)
            Ck.append(Offspring_one)
            Ck.append(Offspring_two)
    
    #Line 11-12:
    Ck_mutated = []
    for index in range(0,len(Ck)):
         mutated_chrom = swap_mutation(Ck[index])                               #### REPLACE by Ck[index] to test if error is in swap_mutation()
         mutated_chrom_next = uniform_mutation(mutated_chrom)
         Ck_mutated.append(mutated_chrom_next)
         print('Iteration',k,'   -->',' Mutations for chromosome',index)
         
    #Collect SE chromosomes from previous population:
    Store_F_Chrom = Store_F_Chrom.sort_values('Fitness',ignore_index=True)
    Remember_previous_pop = Store_F_Chrom[:SE].copy()
      
    #Line 13-14: (based on Line 3-4):
    for index in range(0,NP):
        chromosome = Ck_mutated[index]
        Store_F_Chrom.at[index,'Chromosome'] = chromosome  
        machine_solution_dictionary = create_machine_solutions(chromosome)
        Fitness = result_per_chromosome(machine_solution_dictionary)
        Store_F_Chrom.at[index,'Fitness'] = Fitness
        print('Iteration',k,'   -->',' Evaluate for chromosome',index)
    
    #Sort the Ck_mutated list with corresponding fitness values:   
    Store_F_Chrom = Store_F_Chrom.sort_values('Fitness',ignore_index=True)
    
    #Elitism and immigration:
    #Step 1: Select unique random indexes (Lijst met random getallen bijv. --> [22,45,52,....] met lengte SE)
 #   test = 0
  #  random_indexes_lst = []
   # while test < SE:
   #     number = int(random.uniform(0,len(Store_F_Chrom)))
   #     if number not in random_indexes_lst:
   #         random_indexes_lst.append(number)
   #         test = test + 1
    random_indexes_lst = list(set(range(len(Store_F_Chrom)-SE,len(Store_F_Chrom))))
    #Step 2: Overwrite the rows of the chosen indexes
    for index_prev in range(len(Remember_previous_pop)):
        ###print('orgineel',Store_F_Chrom.at[random_indexes_lst[index_prev],'Chromosome'])
        Store_F_Chrom.at[random_indexes_lst[index_prev],'Chromosome'] = Remember_previous_pop.at[index_prev,'Chromosome']
        Store_F_Chrom.at[random_indexes_lst[index_prev],'Fitness'] = Remember_previous_pop.at[index_prev,'Fitness']
        ###print('aangepast',Store_F_Chrom.at[random_indexes_lst[index_prev],'Chromosome'])
    
    #Step 3: Drop all duplicute rows:
    ### compare = Store_F_Chrom.copy()
    
    for index in range(0,len(Store_F_Chrom)):
        Store_F_Chrom.at[index,'Chromosome'] = tuple(Store_F_Chrom.at[index,'Chromosome'])
    Store_F_Chrom = Store_F_Chrom.drop_duplicates().reset_index(drop = True)
    
    print('DUPLICUTES', NP - len(Store_F_Chrom))
    
    
    while len(Store_F_Chrom) < NP:
        #Create list of chromosomes (as tuples) to check whether the new chromosome is duplicute too.
        lst = []
        for index in range(0,len(Store_F_Chrom)):
            ###print('Last index',index)
            lst.append(Store_F_Chrom.at[index,'Chromosome'])
        new_chrom = tuple(chromosome_generator())
        
        # If new chrom is no duplicute, continue.
        if new_chrom not in lst:  
            #Determine value fitness for new chrom
            machine_solution_dictionary = create_machine_solutions(list(new_chrom))
            Fitness = result_per_chromosome(machine_solution_dictionary)       
            
            #Append the new fitness and chromosome to the Store_F_Chrom table
            new_row = {'Fitness':Fitness,'Chromosome':new_chrom}
            Store_F_Chrom = Store_F_Chrom.append(new_row, ignore_index=True)
        
    #RETURN TUPLES TO LISTS FOR ALL CHROMOSOMES in Store_F_Chrom
    for index in range(0,len(Store_F_Chrom)):
        Store_F_Chrom.at[index,'Chromosome'] = list(Store_F_Chrom.at[index,'Chromosome'])
    
    #Line 16: Sort and store temporarily the best chrom and fitness of this iteration.
    Store_F_Chrom = Store_F_Chrom.sort_values('Fitness',ignore_index=True)
    Best_Fitness_this_iteration = Store_F_Chrom.at[0,'Fitness']
    Best_Chromosome_this_iteration = Store_F_Chrom.at[0,'Chromosome']
    print('Iteration',k,'   -->',' Best fitness of this iteration',Best_Fitness_this_iteration)
    
    ### STORE BEST SOLUTIONS OF ALL ITERATIONS IN TABLE (STILL HAVE TO MAKE IT)
    
    #Line 17-20:
    if Best_Fitness_this_iteration < f_best_Fitness:
        f_best_Fitness = Best_Fitness_this_iteration
        f_best_Chromosome = Best_Chromosome_this_iteration
        best = True
        q = 1
    
    #Line 21-24:
    else:
        best = False
        q = q + 1
    
    print('For generation',k)
    ###print('q =',q,'      ','k = ',k)
    ###print(Store_F_Chrom)
    print('Iteration',k,'   -->',' Timer',(timeit.default_timer()-Start_MH))
    print('______________________________________________________')
    #Line 25-26:
    k = k + 1
    
############################################################################################################################
############################################################################################################################
############################################################################################################################    

    
    














    

