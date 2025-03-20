# THIS FILE IS FROM https://github.com/Ednaordinary/ModelManager

import time

model_manager_path = "../ModelManager/allocation.txt" # This will be different depending on the path to the model manager

def allocate(name):
    allocate = True
    with open(model_manager_path, "r") as allocation_file:
        lines = allocation_file.readlines()
        print(lines)
        if name.strip() in [x[:-1] for x in lines]:
            allocate = False
    if allocate:
        with open(model_manager_path, "a") as allocation_file:
            allocation_file.write(name+"\n")

def deallocate(name):
    with open(model_manager_path, "r") as allocation_file:
        lines = allocation_file.readlines() 
    with open(model_manager_path, "w") as allocation_file:
        for i in lines:
            if i[:-1] != name:
                allocation_file.write(i)

def wait_for_allocation(name):
    last_allocation = None
    while True:
        with open(model_manager_path, "r") as allocation_file:
            lines = allocation_file.readlines()
            if len(lines) != 0:
                if lines[0][:-1] == name:
                    break
                else:
                    if lines[0][:-1] != last_allocation:
                        yield lines[0][:-1]
                        last_allocation = lines[0][:-1]
        time.sleep(0.02)
    return
