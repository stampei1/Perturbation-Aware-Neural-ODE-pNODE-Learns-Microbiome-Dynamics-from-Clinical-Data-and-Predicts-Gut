
with open("args.txt", "w") as f:
    def write_args():
        for replicate in replicates:
            for noise in noises:
                for density in densities:
                    for training_size in training_sizes:
                        f.write(f"--noise {noise} --density {density} --training_size {training_size} --training_time {training_time} --replicate {replicate}")
                        f.write('\n')
    # 2d grid of training size and density
    noises = [0.1]
    densities = [2,3,4,6,8,10]
    training_sizes = [int(2**i) for i in range(8)]
    replicates = [1,2,3,4,5,6,7]
    training_time = 10800
    write_args()

    # 2d grid of noise and training size
    noises = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
    densities = [3]
    training_sizes = [int(2**i) for i in range(8)]
    replicates = [1,2,3,4,5,6,7]
    training_time = 10800
    write_args()

    # varying density
    noises = [0.1]
    densities = [2,3,4,6,8,10]
    training_sizes = [100]
    replicates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    training_time = 10800
    write_args()
    
    # 2d grid of density and noise
    noises = [0, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6, 51.2, 102.4, 204.8]
    densities = [2,3,4,6,8,10]
    training_sizes = [100]
    replicates = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
    training_time = 10800 # 60 minutes
    write_args()

    # varying number of trajectories to zero
    noises = [0.1]
    densities = [3]
    training_sizes = [int(2**i) for i in range(8)]
    replicates = [1,2,3,4,5,6,7]
    training_time = 10800
    write_args()

    # varying noise level at density 3
    noises = [0.2*(3**i) for i in range(8)]
    densities = [3]
    training_sizes = [100]
    replicates = [1,2,3,4,5,6,7]
    training_time = 10800
    write_args()

    # varying noise level at density 10
    noises = [0.2*(3**i) for i in range(8)]
    densities = [10]
    training_sizes = [100]
    replicates = [1,2,3,4,5,6,7]
    training_time = 10800
    write_args()
