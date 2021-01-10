def PH_Chromosome_generator(CSV_file_name):

    ############################################################
    ##############   The Practitioner heuristic   ##############
    ############################################################
    import pandas as pd
    import numpy as np
    import timeit                               #To determine computional time
    from itertools import combinations          #Line 20 (tool assignment) --> Create a list of all possible combinations of values in a list.
    from random import randrange                #Line 20 (tool assignment) --> Choice random combination to take of machine.
    import random
    
    #__________________________________________________________________________________________
    # Start time to determine computational time
    start = timeit.default_timer()
    
    #random.seed(42)
    ############################################################
    ###############        Data selection        ###############
    ############################################################
    
    #_________________________Dataset CSV_file_name____________________________________
    all_data = pd.read_csv(CSV_file_name, names = ['Job i', 'Operation j','Release ij','Process ij', 'Due date d_ij', 'Tool set T_ij', 'Size T_ij'])

    #________________________Parameter selection___________________________
    Capacity_machine = int(all_data.iloc[3][1])                 # Capacity on each machine
    Set_up_time = 1                                             # One hour is used in example 7
    Machines = int(all_data.iloc[1][1])                         # AANTAL MACHINES!!!!!!! DATASET MUST BE IN RIGHT FORMAT!
    Theta_m = 72                                                # 72 hours is 3 days (See section 8 of paper)
    Wd = 1                                                      #(required for objective function)
    Ws = 1                                                      #(required for objective function)


    #________________________Convert dataset___________________________
    raw_input = all_data[6:]
    df = pd.DataFrame(raw_input, columns = ['Job i', 'Operation j','Release ij','Process ij', 'Due date d_ij', 'Tool set T_ij', 'Size T_ij'])
    df['Job i'] = df['Job i'].astype(int)
    df['Operation j'] = df['Operation j'].astype(int)
    df['Release ij'] = df['Release ij'].astype(float)
    df['Process ij'] = df['Process ij'].astype(float)
    df['Due date d_ij'] = df['Due date d_ij'].astype(float)
    df['Tool set T_ij'] = df['Tool set T_ij'].astype(int)
    df['Size T_ij'] = df['Size T_ij'].astype(int)
    #______________________________________________________________________
    

    
    
    ############################################################
    ###############   Further data conversion    ###############
    ############################################################
    
    #__________________________________________________________________________________________
    # Line 1 (Sort EDD)
    df = df.sort_values('Due date d_ij',ignore_index=True) #Including re-indexing the dataframe
    ###T### print (df)
    
    #__________________________________________________________________________________________
    #Create table with unique rows for each Tool. It stores the corresponding size. It is used for line 20 of the PH.
    uni_tool_table = df.copy()
    uni_tool_table = uni_tool_table.drop_duplicates(subset=['Tool set T_ij'])
    uni_tool_table = uni_tool_table.drop(columns=['Job i', 'Operation j','Release ij','Process ij', 'Due date d_ij'])
    uni_tool_table = uni_tool_table.sort_values('Tool set T_ij',ignore_index=True)
    #__________________________________________________________________________________________
    
    
    ###################################################################################
    ###   Starting to assign the initial attached tool sets (Part 1 of example 7)   ###
    ###################################################################################
    
    #List of machines (naming)
    MachineList = list(range(1,Machines+1))
    
    #Tnm_table[] - Columns are the Tnm and Tnm_size (each row contains one machine)
    Tnm = pd.DataFrame(np.zeros((Machines,1)))
    Tnm.columns = ['Tnm_size']
    Tnm.index = MachineList
    Tnm['Tnm_list'] = [ [] for _ in range(Machines) ]
    
    T_not_fit = []   # Review which tools are initially not selected
      
    #__________________________________________________________________________________________
    # Line 2
    index_counter = 0
    for i in df['Job i']:
        j = df['Operation j'][index_counter]
        T_ij = df['Tool set T_ij'][index_counter] # Takes corresponding required tool, T_ij is needed in example 7
        Size_T_ij = df['Size T_ij'][index_counter] # Takes corresponding required tool size, needed in example 7
        index_counter = index_counter + 1
        ###TEST### print('(',i,',',j,') --> T_ij =', T_ij) # This would show all combinations or (i,j) represented in dataframe df
    
    #__________________________________________________________________________________________
    # Checking if tool is already in use
        All_selected_tools = []
        for i_Ast in Tnm['Tnm_list']:
            All_selected_tools = All_selected_tools + i_Ast # Create list of all selected tools
        
        if T_ij not in All_selected_tools:
    
    #__________________________________________________________________________________________
    # Checking on which machines the reviewed tool still fits.
            Fitting_machineList = []
            for i_fit in MachineList:
                if Size_T_ij <= Capacity_machine - Tnm['Tnm_size'][i_fit]:
                    Fitting_machineList.append(i_fit)
                
            if Fitting_machineList == []:
                T_not_fit.append(T_ij)
            
            else: # If there is a machine available on which the part fits continue below.
                
    #__________________________________________________________________________________________
    # Checking which machine has the fewest tools. If multiple (lowest number is chosen)
                smallest_toolset_machine = 0 #Initial value that is not a machine at all.
                smallest_toolset_len = float('inf') #As an initial value it needs to be an unrealistic very large number in terms of tools on a machine.
                for mach in Fitting_machineList:
                    i_len = len(Tnm['Tnm_list'][mach])
                    if i_len < smallest_toolset_len:
                        smallest_toolset_len = i_len
                        smallest_toolset_machine = mach
                
                Tnm['Tnm_list'][smallest_toolset_machine].append(T_ij)
                Tnm.at[smallest_toolset_machine,'Tnm_size']= Tnm.at[smallest_toolset_machine,'Tnm_size'] + Size_T_ij
    
    #__________________________________________________________________________________________
    # Testing the code above (Represents line 2 - 9):
    #
    #print('               The toolset         size')
    #for mach in MachineList:
    #    print('Machine',mach, '-->   ',Tnm['Tnm_list'][mach],'              ',Tnm['Tnm_size'][mach])
    #
    #print('Unselected tools -->', T_not_fit)
    #__________________________________________________________________________________________
    
    
    ########################################################################
    ### Assigning the (i,j) to the machines itself (Part 2 of example 7) ###
    ########################################################################
    
    #Creating a seperate table in which the results can get stored, without affecting the input table.
    results = df.copy()
    # Add rows to the result table
    results['chosen machine'] = len(results['Job i']) * [0]
    results['Tools equip during process'] =  [ [] for _ in range(len(results['Job i'])) ]
    results['start time (A)'] = len(results['Job i']) * [0]
    results['end time (E)'] = len(results['Job i']) * [0]
    results['set up (Z)'] = len(results['Job i']) * [0]
    
    #Available moment for each machine will get stored in the Tnm table as well
    Tnm['Available_t'] = Machines * [0]
    
    #__________________________________________________________________________________________
    # Calling forward the values from df table within the for loop of (i,j)
    index_counter = 0
    for i in df['Job i']:
        j = df['Operation j'][index_counter]
        T_ij = df['Tool set T_ij'][index_counter] # Takes corresponding required tool, T_ij is needed in example 7
        Size_T_ij = df['Size T_ij'][index_counter] # Takes corresponding required tool size, needed in example 7
        Process_ij = df['Process ij'][index_counter] # Takes corresponding process time, needed in example 7
        Release_ij = df['Release ij'][index_counter] # Takes corresponding release time, needed in example 7
    
    #__________________________________________________________________________________________
    # Line 13: What is the fastest machine available  (M_p)
        M_p = 0 #Initial value that is not a machine at all.
        M_p_time = float('inf') #At first an unrealisticly high number is choosen. So that the first option is always earlier.
        for i_fs in MachineList:
            if Tnm['Available_t'][i_fs] < M_p_time:
                M_p_time = Tnm['Available_t'][i_fs]
                M_p = i_fs
    
    #__________________________________________________________________________________________
    # Line 14: Review if a machine already has the required tool attached. (M_t)             (This machine is added to the list called: M_with_req_tool)
        M_with_req_tool = []
        for i_attached in MachineList:
            if T_ij in Tnm['Tnm_list'][i_attached]:
                M_with_req_tool.append(i_attached)
    
        # We will continue if the tool is already placed on a machine. We will identify which of the machines is available first.
        if M_with_req_tool != []:
            M_t = 0 #Initial value that is not a machine at all.
            M_t_time = float('inf') #At first an unrealisticly high number is choosen. So that the first option is always earlier.
            for i_frst in M_with_req_tool:
                if Tnm['Available_t'][i_frst] < M_t_time:
                    M_t_time = Tnm['Available_t'][i_frst]
                    M_t = i_frst
                    ###TEST### print('i_frst -->',i_frst)
        else:
            M_t = 0 #Machine 0 does not exist. So, this means no machine has the tool attached.
        
    #__________________________________________________________________________________________
    # Line 15: If condition
        if M_t != 0:
    #__________________________________________________________________________________________
    # Line 16: If condition + Determining E_M_p and E_M_t values
            if j == 1:
                E_M_p = max(M_p_time,Release_ij)
                E_M_t = max(M_t_time,Release_ij)
            if j > 1:
                E_j_min_one = int(results.loc[(results["Job i"] == i) & (results["Operation j"] == j-1), "end time (E)"]) #Determines when task (i,j-1) was finished. This is required since (i,j) can only start after that timestamp.
                E_M_p = max(M_p_time,E_j_min_one,Release_ij)
                E_M_t = max(M_t_time,E_j_min_one,Release_ij)
            ###TEST### print('(',i,',',j,') -->', 'M_p',M_p, 'M_t',M_t,'  --> E_M_p',E_M_p,'    E_M_t',E_M_t)
                
            if M_p != M_t and E_M_t - E_M_p >= Theta_m:  #Theta_m is a parameter for the tradeoff
    #__________________________________________________________________________________________
    # Line 17-18: So, chose to add new tool + say setup required + add set_up_time to start and endtimes of machines (since setup = idle time)
                Chosen_machine = M_p
                results.at[index_counter,'chosen machine'] = Chosen_machine                         #Store information machine chosen.
                results.at[index_counter,'set up (Z)'] = 1                                          #Store information setup required.
    
                #Additional lines to store data:          
                results.at[index_counter,'start time (A)'] = E_M_p + Set_up_time                    # (i,j) --> Start time (machine is idle during setup)
                Tnm.at[Chosen_machine,'Available_t']  = E_M_p + Process_ij + Set_up_time            # Machine n --> Available time after (i,j)
                results.at[index_counter,'end time (E)']   = E_M_p + Process_ij + Set_up_time       # (i,j) --> End time
    
    #__________________________________________________________________________________________
    # Line 19: Calculate the space that has to be made free for the new tool (so, if new tool size 10. 4 was already free. Req_tool_space is 6)
                Req_tool_space = Size_T_ij - (Capacity_machine - Tnm['Tnm_size'][Chosen_machine])
                ###TEST### print(i,j,T_ij, 'req', Req_tool_space)
                
                
                if Req_tool_space > 0:      #Only than a tool must be taken out!
    #__________________________________________________________________________________________
    # Line 20:         
                    #What tools were already selected
                    eq_set = Tnm['Tnm_list'][Chosen_machine]
                
                    #Create a list of all possible tool combinations
                    comb = []
                    for n in range(1,len(eq_set)+1):
                        for p in combinations(eq_set,n):
                            comb.append(p)
                            
                    #store combinations in dataframe with row to determine sum(size)
                    comb_table = pd.DataFrame(np.zeros((len(comb),1)))
                    comb_table.columns = ['size']
                    comb_table['lists'] = comb
                    
                    
                    #Determine the sum(size) of all combinations of tools
                    for index_comb in range(0,len(comb_table['lists'])):
                        Tool_comb_size = 0
                        for tool in comb_table['lists'][index_comb]:
                            retrieved_size = int(uni_tool_table.loc[(uni_tool_table['Tool set T_ij'] == tool), "Size T_ij"])                
                            Tool_comb_size = Tool_comb_size + retrieved_size
                        comb_table.at[index_comb,'size'] = Tool_comb_size
                
                    #Drop tool combinations that don't make enough space free + reset index
                    comb_table = comb_table[comb_table['size'] >= Req_tool_space].reset_index(drop=True)
                    
    ######2
   #                 #Search for the smallest amount of tools to unselect
    #                smallest_amount = float('inf')
     #               for review_set in comb_table['lists']:
      #                  amount_set = len(review_set)
       #                 if smallest_amount > amount_set:
        #                    smallest_amount = amount_set
         #       
          #          #Drop all rows that would drop more than the minimal number of tools, based on the smallest_amount.
           #         comb_table = comb_table[comb_table['lists'].map(len) == smallest_amount]
            #    
             #       #Randomly chose a combination from the created table.
              #      chosen_index = randrange(0,len(comb_table))
    ######2
                            
    ######1      
    #               #Testing Taking smallest size (no matter the amount of tools this combination of smallest size is) (no randomness)
    #               comb_table = comb_table.sort_values('size',ignore_index=True)
    #               chosen_index = 0 #Smallest size
    ######1
    
    ######0            
                    #Choice a random combination (based on index) that will be used to make space free.
                    chosen_index = randrange(0,len(comb_table))
    ######0 
                
                    #Choice a random combination (based on index) that will be used to make space free.
                    #chosen_index = randrange(0,len(comb_table))
                
                    #This means the following will be droped in line 21:
                    drop_tools_list = comb_table.at[chosen_index,'lists']
                    drop_toolsize = comb_table.at[chosen_index,'size']
                    
                    ###TEST### print('pre list',Tnm.at[Chosen_machine,'Tnm_list'], 'pre size',Tnm.at[Chosen_machine,'Tnm_size'])
                    
                    # Dropping the tools
                    for drop_tool in drop_tools_list:
                        Tnm.at[Chosen_machine,'Tnm_list'].remove(drop_tool)
                    Tnm.at[Chosen_machine,'Tnm_size'] = Tnm.at[Chosen_machine,'Tnm_size'] - drop_toolsize   
                
                    ###TEST### print('drop list',drop_tools_list, 'drop size', drop_toolsize)
                    ###TEST### print('---')
                    ###TEST### print('mid list',Tnm.at[Chosen_machine,'Tnm_list'], 'mid size',Tnm.at[Chosen_machine,'Tnm_size'])
    #__________________________________________________________________________________________
    # Line 21: After space was made on the machine the newly required tool is added.   
                Tnm.at[Chosen_machine,'Tnm_list'].append(T_ij)
                Tnm.at[Chosen_machine,'Tnm_size'] = Tnm.at[Chosen_machine,'Tnm_size'] + Size_T_ij
                    
                #Store additional information:
                results.at[index_counter,'Tools equip during process'] = Tnm.at[Chosen_machine,'Tnm_list'].copy()
                    
                ###TEST### print('add list',T_ij, 'add size',Size_T_ij)
                ###TEST### print('---')
                ###TEST### print('after list',Tnm.at[Chosen_machine,'Tnm_list'], 'after size',Tnm.at[Chosen_machine,'Tnm_size'])
                ###TEST### print('#######################')
                
    
    #__________________________________________________________________________________________
    # Line 22-23-24: Chose to use a machine with the tool already attached + No setup required.:
            else:
                Chosen_machine = M_t
                results.at[index_counter,'chosen machine'] = Chosen_machine         #Store information machine chosen.
                results.at[index_counter,'set up (Z)'] = 0                          #Store information setup required.
                
                #Additional lines to store data:          
                results.at[index_counter,'start time (A)'] = E_M_t                  # (i,j) --> Start time (machine is idle during setup)
                Tnm.at[Chosen_machine,'Available_t']  = E_M_t + Process_ij          # Machine n --> Available time after (i,j)
                results.at[index_counter,'end time (E)']   = E_M_t + Process_ij     # (i,j) --> End time
                results.at[index_counter,'Tools equip during process'] = Tnm['Tnm_list'][Chosen_machine].copy()
    #__________________________________________________________________________________________
    # Line 25-26-27 (beware that the if j  == 1 and j > 1 has to be here, next to the code of line 17-21)                 
        else:
            #Obtaining missing values for line 17-21:
            if j == 1:
                E_M_p = max(M_p_time,Release_ij)
            if j > 1:
                E_j_min_one = int(results.loc[(results["Job i"] == i) & (results["Operation j"] == j-1), "end time (E)"]) #Determines when task (i,j-1) was finished. This is required since (i,j) can only start after that timestamp.
                E_M_p = max(M_p_time,E_j_min_one,Release_ij)        
            
            #Line 17-18:
            Chosen_machine = M_p
            results.at[index_counter,'chosen machine'] = Chosen_machine                         #Store information machine chosen.
            results.at[index_counter,'set up (Z)'] = 1                                          #Store information setup required.
    
            #Additional lines to store data:          
            results.at[index_counter,'start time (A)'] = E_M_p + Set_up_time                    # (i,j) --> Start time (machine is idle during setup)
            Tnm.at[Chosen_machine,'Available_t']  = E_M_p + Process_ij + Set_up_time            # Machine n --> Available time after (i,j)
            results.at[index_counter,'end time (E)']   = E_M_p + Process_ij + Set_up_time       # (i,j) --> End time
            
            #Line 19-20-21:
            # PASTE BELOW BETWEEN THE ---------- lines !!!!
    #----------------------------------------------------------------------------
    # Line 19: Calculate the space that has to be made free for the new tool (so, if new tool size 10. 4 was already free. Req_tool_space is 6)
            Req_tool_space = Size_T_ij - (Capacity_machine - Tnm['Tnm_size'][Chosen_machine])
            
                
                
            if Req_tool_space > 0:      #Only than a tool must be taken out!
                ###TEST### print(i,j,T_ij, 'req', Req_tool_space)
    #__________________________________________________________________________________________
    # Line 20:         
                #What tools were already selected
                eq_set = Tnm['Tnm_list'][Chosen_machine]
                
                #Create a list of all possible tool combinations
                comb = []
                for n in range(1,len(eq_set)+1):
                    for p in combinations(eq_set,n):
                        comb.append(p)
                            
                #store combinations in dataframe with row to determine sum(size)
                comb_table = pd.DataFrame(np.zeros((len(comb),1)))
                comb_table.columns = ['size']
                comb_table['lists'] = comb
                
                
                #Determine the sum(size) of all combinations of tools
                for index_comb in range(0,len(comb_table['lists'])):
                    Tool_comb_size = 0
                    for tool in comb_table['lists'][index_comb]:
                        retrieved_size = int(uni_tool_table.loc[(uni_tool_table['Tool set T_ij'] == tool), "Size T_ij"])                
                        Tool_comb_size = Tool_comb_size + retrieved_size
                    comb_table.at[index_comb,'size'] = Tool_comb_size
            
                #Drop tool combinations that don't make enough space free + reset index
                comb_table = comb_table[comb_table['size'] >= Req_tool_space].reset_index(drop=True)
    
    ######2
   #             #Search for the smallest amount of tools to unselect
    #            smallest_amount = float('inf')
     #           for review_set in comb_table['lists']:
      #              amount_set = len(review_set)
       #             if smallest_amount > amount_set:
        #                smallest_amount = amount_set
         #       
          #      #Drop all rows that would drop more than the minimal number of tools, based on the smallest_amount.
           #     comb_table = comb_table[comb_table['lists'].map(len) == smallest_amount]
            #    
             #   #Randomly chose a combination from the created table.
              #  chosen_index = randrange(0,len(comb_table))
    ######2
                            
    ######1      
    #            #Testing Taking smallest size (no matter the amount of tools this combination of smallest size is) (no randomness)
    #            #comb_table = comb_table.sort_values('size',ignore_index=True)
    #            #chosen_index = 0 #Smallest size
    ######1
    
    ######0            
                #Choice a random combination (based on index) that will be used to make space free.
                chosen_index = randrange(0,len(comb_table))
    ######0            
                #This means the following will be droped in line 21:
                drop_tools_list = comb_table.at[chosen_index,'lists']
                drop_toolsize = comb_table.at[chosen_index,'size']
                    
                ###TEST### print('pre list',Tnm.at[Chosen_machine,'Tnm_list'], 'pre size',Tnm.at[Chosen_machine,'Tnm_size'])
                    
                # Dropping the tools
                for drop_tool in drop_tools_list:
                    Tnm.at[Chosen_machine,'Tnm_list'].remove(drop_tool)
                Tnm.at[Chosen_machine,'Tnm_size'] = Tnm.at[Chosen_machine,'Tnm_size'] - drop_toolsize   
                
                ###TEST### print('drop list',drop_tools_list, 'drop size', drop_toolsize)
                ###TEST### print('---')
                ###TEST### print('mid list',Tnm.at[Chosen_machine,'Tnm_list'], 'mid size',Tnm.at[Chosen_machine,'Tnm_size'])
    #__________________________________________________________________________________________
    # Line 21: After space was made on the machine the newly required tool is added.   
            Tnm.at[Chosen_machine,'Tnm_list'].append(T_ij)
            Tnm.at[Chosen_machine,'Tnm_size'] = Tnm.at[Chosen_machine,'Tnm_size'] + Size_T_ij
            
            #Store additional information:
            results.at[index_counter,'Tools equip during process'] = Tnm.at[Chosen_machine,'Tnm_list'].copy()
            
            ###TEST### print('add list',T_ij, 'add size',Size_T_ij)
            ###TEST### print('---')
            ###TEST### print('after list',Tnm.at[Chosen_machine,'Tnm_list'], 'after size',Tnm.at[Chosen_machine,'Tnm_size'])
            ###TEST### print('#######################')
    #----------------------------------------------------------------------------
    
        
    #__________________________________________________________________________________________
    #Line 28-29-30-31-32-33-34 were included in the code above. It looked for every (i,j) what the end time at the corresponding machine was.
    #The stored information can be retrieved from the 'result' table in column 'End time (E)'.
    
    #__________________________________________________________________________________________
        # At the end of the for loop the index counter is raised by 1
        index_counter = index_counter + 1
    
    #__________________________________________________________________________________________
    # Line 35 (Delta D calculations):
    Indexlist = list(range(0,len(results['Job i'])))
    results['Delta D'] = len(results['Job i']) * [0]
    for all_rows in Indexlist:
        results.at[all_rows,'Delta D']   = max(0,results.at[all_rows,'end time (E)']-results.at[all_rows,'Due date d_ij'])
    #__________________________________________________________________________________________
    
    
    ######################################################
    ###                     Results                    ###
    ######################################################
    #Overall performance measures
    print('_______________________________')
    print('OBJECTIVE VALUE =',Wd*sum(results['Delta D'])+Ws*sum(results['set up (Z)']))
    print('         Total tardiness -->',sum(results['Delta D']))
    print('         Total setup    -->',sum(results['set up (Z)']))
    print('_______________________________')
    
    #The sequence on every machine.
    for m in MachineList:
        sequence = str()
        for index in Indexlist:
            if results.at[index,'chosen machine'] == m:
                sequence = sequence + str('(') + str(results.at[index,'Job i']) + str(',') + str(results.at[index,'Operation j']) + str(')')+ str('>')
    #    print('Machine ',m,'-->',sequence[:-1])
    #print('_______________________________')
    
    #__________________________________________________________________________________________
    # End time to determine computational time
    stop = timeit.default_timer()
    total_time = round(stop - start,4)
    print('Computational time: ', total_time )
    
    
    ########################################################################
    ####################### Output for Math heuristic ######################
    ########################################################################
    
    output = results.copy()
    output = output.sort_values('start time (A)',ignore_index=True) #Including re-indexing the dataframe
    chrom_PH = []
    for index in range(0,len(output['start time (A)'])):
        gene = (  output.at[index,'Job i']    ,   output.at[index,'chosen machine']    )
        chrom_PH.append(gene)
    ###TEST### print(chrom_PH)
    return chrom_PH                  
    

#Testing output!
x = PH_Chromosome_generator('6M140.csv')
#print('CHROMOSOOM PH_TST', x)
#print('len',len(x))


