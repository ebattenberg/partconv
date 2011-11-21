#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <errno.h>

#include "fftw3.h"
#include "buffers.h"
#include "sndtools.h"

#include "partconvMulti.h"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

#define MEASURE_COMPUTE_TIME
#define _DEBUG_


/* PartConvMulti methods */

int PartConvMulti::setup( const char* config_file_name, int max_threads_per_level, int max_level0_threads, double min_work)
{
    int ret;
    FILE *config_file;
    char ir_filename[256]; // impulse response filename

    config_file = fopen(config_file_name, "r");

    if (config_file == NULL) {
        fprintf(stderr, "Unable to open config file %s\n", config_file_name);
        exit(-1);
    }

    ret = fscanf(config_file, "%d %d %d %d %d %d\n",&FS, &buffer_size, &numInputChannels, &numOutputChannels, &numPCs, &numRepeats);
    if (ret != 6) {
        fprintf(stderr, "Error reading config file!\n");
        exit(-1);
    }

    if (verbosity > 0)
    {
        printf("\nnumInputChannels = %d\n", numInputChannels);
        printf("numOutputChannels = %d\n", numOutputChannels);
        printf("numPCs = %d\n", numPCs);
        printf("numRepeats = %d\n\n", numRepeats);
    }
    else
        printf("\nsetting up PartConvMulti\n\n");


    int **N = new int*[numPCs*numRepeats];
    int **num_parts = new int*[numPCs*numRepeats];
    int *M = new int[numPCs*numRepeats];

    inputChannel = new int[numPCs*numRepeats];
    outputChannel = new int[numPCs*numRepeats];
    muteChannel = new int[numPCs*numRepeats];
    scaleFactor = new float[numPCs*numRepeats];
    
    output_mix = new float[buffer_size*numOutputChannels];
    input_mix = new float[buffer_size*numInputChannels];

    pc = new PartConvFilter[numPCs*numRepeats];
    
    max_block_size = 0;

    char **filenames = new char*[numPCs*numRepeats];

    for (int i=0; i<numPCs; i++)
    {
        ret = fscanf(config_file, "%s %d %d %d %f %d\n", ir_filename, &muteChannel[i], &inputChannel[i], &outputChannel[i], &scaleFactor[i], &M[i]);
        if (ret != 6) {
            fprintf(stderr, "Error reading config file! 2\n");
            exit(-1);
        }
        filenames[i] = new char[strlen(ir_filename)];
        memcpy(filenames[i], ir_filename, strlen(ir_filename)*sizeof(char));
        if (verbosity > 0)
        {
            printf("filename = %s, muteChannel = %d, inputChannel = %d, outputChannel = %d, scaleFactor = %0.2f, M = %d\n",
                    ir_filename, muteChannel[i], inputChannel[i], outputChannel[i], scaleFactor[i], M[i]);
        }

        N[i] = new int[M[i]]();
        num_parts[i] = new int[M[i]]();

        for (int j=0; j<M[i]; j++) {
            ret = fscanf(config_file, "%d %d", &N[i][j], &num_parts[i][j]);

            if (ret < 1 || ret > 2) {
                fprintf(stderr, "Error reading config file! 3\n");
                exit(-1);
            }

            if (max_block_size < N[i][j])
                max_block_size = N[i][j];

            if (j == 0)
            {
                if (N[i][0] != buffer_size)
                {
                    fprintf(stderr, "line %u: first block size must equal buffer_size\n",i+1);
                    exit(-1);
                }
            }

            if (j == M[i]-1)
                num_parts[i][j] = 0;

            if (verbosity > 0)
                printf("%d %d ", N[i][j], num_parts[i][j]);
        }
        if (verbosity > 0)
            printf("\n");

        for (int j = 1; j < numRepeats; j++)
        {
            N[i+j*numPCs] = new int[M[i]];
            num_parts[i+j*numPCs] = new int[M[i]];
            filenames[i+j*numPCs] = new char[strlen(ir_filename)];

            M[i+j*numPCs] = M[i];
            memcpy(N[i+j*numPCs],N[i],M[i]*sizeof(int));
            memcpy(num_parts[i+j*numPCs],num_parts[i],M[i]*sizeof(int));
            memcpy(filenames[i+j*numPCs], ir_filename, strlen(ir_filename)*sizeof(char));

            muteChannel[i+j*numPCs] = 1;   //mute the repeat pc's
            inputChannel[i+j*numPCs] = inputChannel[i];
            outputChannel[i+j*numPCs] = outputChannel[i];
            scaleFactor[i+j*numPCs] = scaleFactor[i];
        }
    }

    fclose(config_file);
    numPCs *= numRepeats;

    int temp_num_levels = log2i(max_block_size) - log2i(buffer_size) + 1;
    unsigned temp_level_size[temp_num_levels];

    for (int i = 0; i < temp_num_levels; i++)
        temp_level_size[i] = 1 << i; 

    int chunks_in_levels[temp_num_levels];
    bool which_levels[numPCs][temp_num_levels];

    // find which levels are used by different pc instances
    if (verbosity > 1)
        printf("\nLevel Occurrence\n");
    num_levels = 0;
    for (int i = 0; i < temp_num_levels; i++)
    {
        chunks_in_levels[i] = 0;
        const int block_size = buffer_size * (1 << i);
        for (int j = 0; j < numPCs; j++)
        {
            which_levels[j][i] = false;

            for (int k = 0; k < M[j]; k++)
            {
                if (N[j][k] == block_size)
                {
                    which_levels[j][i] = true;
                    chunks_in_levels[i]++;
                }
                else if (N[j][k] > block_size)
                    break;
                    
            }
            if (verbosity > 1)
                printf("%d",which_levels[j][i]);
        }
        if (chunks_in_levels[i] > 0)
            num_levels++;
        if (verbosity > 1)
        {
            printf(" = %u",chunks_in_levels[i]); 
            printf("\n");
        }
    }
    if (verbosity > 1)
        printf("\n");


    
    level_counter = new unsigned[num_levels]; 
    level_size = new unsigned[num_levels];
    level_map = new int*[numPCs];
    level_map[0] = new int[numPCs*num_levels];
    for (int i = 1; i < numPCs; i++)
        level_map[i] = level_map[0] + i*num_levels;

    if (verbosity > 1)
        printf("\nLevel Mapping\n");
    int ind = 0;
    for (int i = 0; i < num_levels; i++)
    {
        while (1)
        {
            if (chunks_in_levels[ind] > 0)
            {
                level_size[i] = temp_level_size[ind];
                level_counter[i] = level_size[i] - 1;
                const int block_size = buffer_size * level_size[i];
                if (verbosity > 1)
                    printf("%6u ", block_size);

                for (int j = 0; j < numPCs; j++)
                {
                    level_map[j][i] = -1;
                    for (int k = 0; k < M[j]; k++)
                    {
                        if (N[j][k] == block_size)
                        {
                            level_map[j][i] = k;
                        }
                        else if (N[j][k] > block_size)
                            break;

                    }
                    if (verbosity > 1)
                    {
                        if (level_map[j][i] > -1)
                            printf("%1d  ",level_map[j][i]);
                        else
                            printf("%3s","\\  ");
                    }
                }

                ind++;
                break;
            }
            else
            {
                ind++;
            }
        }
        if (verbosity > 1)
            printf("\n");
    }
    if (verbosity > 1)
        printf("\n");


    double exec_time[numPCs][num_levels];
    memset(exec_time,0,sizeof(exec_time));


    if (FILE* wisdomfile = fopen("wisdom.wis","r"))
    {
        fftwf_import_wisdom_from_file(wisdomfile);
        fclose(wisdomfile);
    }

    for (int i = 0; i < numPCs; i++)
    {

        if (verbosity > 2)
        {
            pc[i].verbosity = 1;
        }
        else
        {
            pc[i].verbosity = 0;
        }


        Vector impulse_response; 
        int err = readImpulseResponse(impulse_response,filenames[i]);
        if (err)
        {
            fprintf(stderr, "unable to open file: %s\n",filenames[i]);
            //goto error; //free stuff before returning
            //memory leak
            exit(-1);
        }
        const int fftwOptLevel = 2; //FFTW_PATIENT

        err = pc[i].setup(M[i],N[i],num_parts[i],&impulse_response,fftwOptLevel);
        if (err) {
            printf("setup PartConv %u: failed\n",i);
            exit(-1);
        }

        delete[] N[i];
        delete[] num_parts[i];
        delete[] filenames[i];

#ifdef MEASURE_COMPUTE_TIME
        const int trials = 100;
        const int pollute_every = 3;  
        const float max_time = 0.1;
        if (verbosity > 1)
            printf("\nPartConv %u\n",i);
        for (int j = 0; j < num_levels; j++)
        {
            const int L = level_map[i][j];
            if (L != -1)
            {
                double t;
                if (j == 0)
                {
                    t = pc[i].bench(L,trials,pollute_every,max_time);
                    if (verbosity > 1)
                    {
                        printf("%2u: %6u x %-4u: %4.2fms (%5.2f%%) [%6.2f%%]\n",j,pc[i].N[L],pc[i].num_parts[L],
                                t*1000, 100*t/((double)pc[i].N[L]/FS), 100*t/((double)pc[i].N[0]/FS));
                    }
                }
                else 
                    t = (double)pc[i].N[j]*pc[i].num_parts[j];

                exec_time[i][j] = t;
            }
        }
#else
        for (int j = 0; j < num_levels; j++)
        {
            const int L = level_map[i][j];
            if (L != -1)
            {
                double t = (double)pc[i].N[j]*pc[i].num_parts[j];
                exec_time[i][j] = t;
            }
        }
#endif

    }
    if (FILE* wisdomfile = fopen("wisdom.wis","w"))
    {
        fftwf_export_wisdom_to_file(wisdomfile);
        fclose(wisdomfile);
    }
    delete[] N;
    delete[] M;
    delete[] num_parts;
    delete[] filenames;

    // divide up work amongst threads
    // for now we cluster adjacent work units,
    // could get more uniform work division if lifting this constraint
    threads_in_level = new int[num_levels](); 
    workList = new WorkList**[num_levels];
    workList[0] = new WorkList*[num_levels*max_threads_per_level](); //init 0's (NULL's)
    for (int i = 1; i < num_levels; i++)
        workList[i] = workList[0] + i*max_threads_per_level;

    double work_level_sum[num_levels];
    for (int l = 0; l < num_levels; l++)
    {
        work_level_sum[l] = 0;
        double work_target = 0;

#ifdef MEASURE_COMPUTE_TIME
        for (int i = 0; i < numPCs; i++)
            work_target += exec_time[i][l];
        work_target /= max_threads_per_level;
        if (work_target < min_work)
            work_target = min_work;
#else
        for(int i = 0; i < numPCs; i++)
        {
            if (level_map[i][l] != -1)
                work_target += exec_time[i][l];
        }
        work_target /= max_threads_per_level;
#endif

        if (verbosity > 1)
            printf("%u: work_target = %f\n",l,work_target);

        int unit = 0;
        double work_sum = 0;
        for (int indPC = 0; indPC < numPCs; indPC++)
        {
            if (level_map[indPC][l] < 0)
                continue;

            if (work_sum >= work_target)
            {
                work_sum = 0;
                unit++;
                if (unit == max_threads_per_level)
                {
                    fprintf(stderr,"work partitioning error\n");
                    exit(-1);
                }
            }

            // add work unit to linked list
            workList[l][unit] = new WorkList(workList[l][unit]);
            const double t = exec_time[indPC][l];

            workList[l][unit]->arg = level_map[indPC][l];
            workList[l][unit]->ind = indPC;
            workList[l][unit]->obj = &pc[indPC];
            workList[l][unit]->time = exec_time[indPC][l];
            work_sum += t;
            work_level_sum[l] += exec_time[indPC][l];
            if (verbosity > 1)
                printf("level: %u, unit: %u, indPC: %u, new_work: %f, work_sum: %f\n",l,unit,indPC,t,work_sum);
        }
        threads_in_level[l] = unit+1;

        if (verbosity > 1)
            printf("threads total: %u\n",threads_in_level[l]);
    }

    // Level 0 stuff
    total_level0_work_units = threads_in_level[0];
    threads_in_level[0] = min( total_level0_work_units - 1, max_level0_threads );
    first_thread_level = (threads_in_level[0] == 0); // 0 if no L0 threads
    // -----



    if (verbosity > 2) //thread assignment debugging info
    {
        for (int l = 0; l < num_levels; l++)
        {
            for (int thr = 0; thr < threads_in_level[l]; thr++)
            {
                WorkList *curr = workList[l][thr];
                printf("\nlevel: %u, thr: %u\n",l,thr);
                while(curr != NULL)
                {
                    printf("arg: %u, ind: %u\n",curr->arg,curr->ind);
                    curr = curr->next;
                }
            }
            if (l == 0)
            {
                for (int unit = threads_in_level[l]; unit < total_level0_work_units; unit++)
                {
                    WorkList *curr = workList[l][unit];
                    printf("\nlevel: %u, unit: %u,",l,unit);
                    printf(" runs in callback: \n");
                    while(curr != NULL)
                    {
                        printf("arg: %u, ind: %u\n",curr->arg,curr->ind);
                        curr = curr->next;
                    }
                }
            }

        }

    }


    if (verbosity > 1) //print work per level
    {
        printf("\nWorkload per level:\n");
        double work_total_sum = 0;

        for (int i = 0; i < num_levels; i++)
        {
            double level_percentage = 100*work_level_sum[i]/((double)level_size[i]*buffer_size/FS);
            work_total_sum += level_percentage;
            printf("Level %2u: %6.3f\n", i, level_percentage);
        }
        printf("Total   : %6.3f\n\n", work_total_sum);
    }
    







    // setup threading stuff ----------------------------------------------

    terminate_flag = 0;

    // worker_thread[level][thread]
    worker_thread = new pthread_t*[num_levels];
    worker_thread[0] = new pthread_t[num_levels*max_threads_per_level];
    for (int i = 1; i < num_levels; i++)
        worker_thread[i] = worker_thread[0] + i*max_threads_per_level;

    workerThreadData = new WorkerThreadData*[num_levels];
    workerThreadData[0] = new WorkerThreadData[num_levels*max_threads_per_level];
    for (int i = 1; i < num_levels; i++)
        workerThreadData[i] = workerThreadData[0] + i*max_threads_per_level;

    worker_state = new int*[num_levels]; 
    worker_state[0] = new int[num_levels*max_threads_per_level](); //init 0's
    for (int i = 1; i < num_levels; i++)
        worker_state[i] = worker_state[0] + i*max_threads_per_level;
    
    thread_counter = new int[num_levels](); //init 0
    sync_target = new int[num_levels](); //init 0
    worker_cond = new pthread_cond_t[num_levels];
    worker_mutex = new pthread_mutex_t[num_levels];
    main_cond = new pthread_cond_t[num_levels];
    main_mutex = new pthread_mutex_t[num_levels];
    pthread_attr_t thread_attr[num_levels];


    for (int i = first_thread_level; i < num_levels; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            sync_target[i] += threads_in_level[j];
        }
    }

    for (int i = first_thread_level; i < num_levels; i++)
    {
        pthread_cond_init(&worker_cond[i],NULL);
        pthread_mutex_init(&worker_mutex[i],NULL);
        pthread_cond_init(&main_cond[i],NULL);
        pthread_mutex_init(&main_mutex[i],NULL);
    }


#if !defined(__APPLE__) && (defined (PIN_BY_INSTANCE) || defined (PIN_BY_LEVEL))
    // allocate cpuset for pthread_setaffinity_np() below
    cpu_set_t cpuset;
    const unsigned num_cpus = get_nprocs(); //sys/sysinfo.h
    if (CPU_SETSIZE(cpuset) < num_cpus)
    {
        fprintf(stderr, "CpuSet error: must dynamically allocate a bigger cpuset\n");
        exit(-1);
    }
#endif

    for (int i = first_thread_level; i < num_levels; i++)
    {
        struct sched_param thread_param;
        pthread_attr_init(&thread_attr[i]);
        pthread_attr_setdetachstate(&thread_attr[i], PTHREAD_CREATE_JOINABLE);
        pthread_attr_setschedpolicy(&thread_attr[i], SCHED_FIFO);

        pthread_attr_getschedparam(&thread_attr[i], &thread_param);
#ifdef __APPLE__
        const int actual_max_priority = 63;
        //thread_param.sched_priority = sched_get_priority_max(SCHED_FIFO) - i + 15;
        thread_param.sched_priority = actual_max_priority - i;
#else
        //thread_param.sched_priority = sched_get_priority_max(SCHED_FIFO) - i - 1;
        thread_param.sched_priority = 80 - i;
#endif
        //printf("thread %u: priority = %u\n",i,thread_param.sched_priority);
        pthread_attr_setschedparam(&thread_attr[i], &thread_param);
        pthread_attr_setinheritsched(&thread_attr[i], PTHREAD_EXPLICIT_SCHED);


        //printf("%u: level_time = %f\n",i,level_time*1000);

        for (int j = 0; j < threads_in_level[i]; j++)
        {
            workerThreadData[i][j].This = this;
            workerThreadData[i][j].levelNum = i;
            workerThreadData[i][j].threadNum = j;

            worker_state[i][j] = 1;
#ifndef __APPLE__
#ifdef PIN_BY_INSTANCE
            const int core = j % num_cpus;
#endif
#ifdef PIN_BY_LEVEL
            const int core = i % num_cpus;
#endif
#if defined(PIN_BY_INSTANCE) || defined(PIN_BY_LEVEL)
            CPU_ZERO(&cpuset);
            CPU_SET(core, &cpuset);
            pthread_attr_setaffinity_np(&thread_attr[i], sizeof(cpu_set_t), &cpuset);
#endif
#endif

            int err = pthread_create(&worker_thread[i][j], &thread_attr[i], workerThreadEntry, (void*)&workerThreadData[i][j]);
            if ( err != 0) 
            {
                printf("pthread_create error: %d\n",err); 
                if (err == EPERM)
                    printf("EPERM\n");
                exit(-1);
            }


#if defined(__APPLE__) && defined(MEASURE_COMPUTE_TIME)
            if (i == 0)
            {
                const int bus_speed = get_bus_speed();
                const float level_time = (float)level_size[i]*buffer_size/FS;
                const float callback_time = (float)buffer_size/FS;
                double work_sum = 0;
                WorkList *curr = workList[i][j];
                while (curr != NULL)
                {
                    work_sum += curr->time;
                    curr = curr->next;
                }
                if (verbosity > 0)
                    printf("%u.%u: work_sum = %f\n",i,j,work_sum*1000);

                thread_time_constraint_policy_data_t time_policy;
                time_policy.period = (uint32_t)bus_speed * .93985 * level_time;
                //int comp_max = min(5E5,bus_speed*((double).93985*N[0]/FS/M));
                const uint32_t comp_max = 5E7;
                const uint32_t comp_min = 5E4;
                //time_policy.computation = (uint32_t)min(comp_max,max(comp_min,(work_sum * bus_speed))); //5E4/1064E6=0.05ms
                time_policy.computation = (uint32_t)(work_sum * bus_speed);
                time_policy.computation = max( time_policy.computation, comp_min); //5E4/1064E6=0.05ms
                time_policy.computation = min( time_policy.computation, comp_max); //5E4/1064E6=0.05ms
                time_policy.constraint = (uint32_t)time_policy.period * 0.7;  // would like to leave cushion
                time_policy.preemptible = 1;

                if (work_sum < callback_time)
                {
                    err = thread_policy_set(
                            pthread_mach_thread_np(worker_thread[i][j]),
                            THREAD_TIME_CONSTRAINT_POLICY,
                            (thread_policy_t)&time_policy,
                            THREAD_TIME_CONSTRAINT_POLICY_COUNT);
                    if (err != KERN_SUCCESS)
                    {
                        printf("mach set policy error: %d\n",err); 
                        printf("not able to set time constraint policy for level: %u.%u\n",i,j);
                        printf("%u.%u mach cycles: %u %u %u %u %u\n",
                                i,j,
                                time_policy.period,
                                time_policy.computation,
                                time_policy.constraint,
                                time_policy.preemptible,
                                get_bus_speed()
                              );
                        printf("%u.%u mach time: %g %g %g %u %u\n",
                                i,j,
                                time_policy.period/(float)bus_speed,
                                time_policy.computation/(float)bus_speed,
                                time_policy.constraint/(float)bus_speed,
                                time_policy.preemptible,
                                get_bus_speed()
                              );
                    }
                }

#ifdef _DEBUG_
                // check policy params

                mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
                boolean_t get_default = 0;
                err = thread_policy_get(
                        pthread_mach_thread_np(worker_thread[i][j]),
                        THREAD_TIME_CONSTRAINT_POLICY,
                        (thread_policy_t)&time_policy,
                        &count,
                        &get_default);
                if ( err != KERN_SUCCESS) 
                {
                    printf("mach get policy error: %d (L=%u)\n",err,i); 
                    //exit(-1);
                }
                else
                {
                    printf("%u mach time: %u %u %u %u %u %u\n",
                            i,
                            time_policy.period,
                            time_policy.computation,
                            time_policy.constraint,
                            time_policy.preemptible,
                            get_bus_speed(),
                            get_default
                          );
                }
                get_default = 0;
                count = THREAD_PRECEDENCE_POLICY_COUNT;
                thread_precedence_policy_data_t prec_policy;
                err = thread_policy_get(
                        pthread_mach_thread_np(worker_thread[i][j]),
                        THREAD_PRECEDENCE_POLICY,
                        (thread_policy_t)&prec_policy,
                        &count,
                        &get_default);
                if ( err != KERN_SUCCESS) 
                {
                    printf("mach get precedence error: %d (L=%u)\n",err,i); 
                    //exit(-1);
                }
                else
                {
                    printf("%u mach prec: %d %d\n", i, prec_policy.importance, get_default);
                }
#endif
            }
#endif
        }
        pthread_attr_destroy(&thread_attr[i]);

    }

    printf("created threads\n");
    printf("first_thread_level: %u\n",first_thread_level);

    int end_sync_depth = num_levels-1;
    pthread_mutex_lock(&main_mutex[end_sync_depth]);
    while( thread_counter[end_sync_depth] != sync_target[end_sync_depth])
        pthread_cond_wait(&main_cond[end_sync_depth],&main_mutex[end_sync_depth]);
    pthread_mutex_unlock(&main_mutex[end_sync_depth]);


    printf("Total PartConv instances   : %d\n\n", numPCs);
    printf("buffer_size: %u\n",buffer_size);
    printf("max_block_size: %u\n\n",max_block_size);
    frameNum = 0;
    lastFrame = 0;
    doneWaiter = NULL;





    return 0;
}

int PartConvMulti::cleanup(void)
{
    // join worker threads
    __sync_add_and_fetch(&terminate_flag, 1);
    for (int i = first_thread_level; i < num_levels; i++)
    {
        for (int j = 0; j < threads_in_level[i]; j++)
            worker_state[i][j] = -1;

    }
    for (int i = first_thread_level; i < num_levels; i++)
        pthread_cond_broadcast(&worker_cond[i]);

    for (int i = first_thread_level; i < num_levels; i++)
    {
        for (int j = 0; j < threads_in_level[i]; j++)
            pthread_join(worker_thread[i][j],NULL);
    }
        
    for (int i = 0; i < numPCs; i++)
    {
        pc[i].cleanup();
    }




    delete[] inputChannel;
    delete[] outputChannel;
    delete[] muteChannel;
    delete[] scaleFactor;
    delete[] pc;

    delete[] input_mix;
    delete[] output_mix;

    delete[] level_counter; 
    delete[] level_size;
    delete[] level_map[0];
    delete[] level_map;


    for (int l = 0; l < num_levels; l++)
    {
        for (int thr = 0; thr < threads_in_level[l]; thr++)
        {
            WorkList *curr = workList[l][thr];
            WorkList *next;
            while(curr != NULL)
            {
                next = curr->next;
                delete curr;
                curr = next;
            }
        }

    }

    delete[] threads_in_level;
    delete[] sync_target;
    delete[] thread_counter;
    delete[] workList[0];
    delete[] workList;

    for (int i = first_thread_level; i < num_levels; i++)
    {
        pthread_cond_destroy(&worker_cond[i]);
        pthread_mutex_destroy(&worker_mutex[i]);
        pthread_cond_destroy(&main_cond[i]);
        pthread_mutex_destroy(&main_mutex[i]);
    }
    delete[] worker_thread[0];
    delete[] worker_thread;
    delete[] workerThreadData[0];
    delete[] workerThreadData;
    delete[] worker_state[0];
    delete[] worker_state;
    delete[] worker_cond;
    delete[] worker_mutex;
    delete[] main_cond;
    delete[] main_mutex;


    return 0;
}

void PartConvMulti::reset()
{ 
    //reset all internal buffers
    // do we need to reset any other state?

    for (int i = 0; i < numPCs; i++)
        pc[i].reset();
}

int PartConvMulti::run(float* const output, const float* const input, const int frameCount) 
{
    if (frameCount != buffer_size)
    {
        fprintf(stderr,"frameCount (%u) != buffer_size (%u)\n",frameCount,buffer_size);
        exit(-1);
    }

    // which levels need to sync?  -------------------
    int start_sync_depth = 0;
    int end_sync_depth = 0;

    for (int l = 0; l < num_levels; l++)
    {
        level_counter[l]++;
        if (level_counter[l] == level_size[l])
        {
            level_counter[l] = 0;
            start_sync_depth = l;
        }
        else if (level_counter[l] == level_size[l]-1)
            end_sync_depth = l;
    }

    // deinterleave input
    for (int i = 0; i < numPCs; i++)
    {
        const int chan = inputChannel[i];
        const float scale = scaleFactor[i];

        for (int j = 0; j < frameCount; j++)
            pc[i].input_mix[j] = input[chan + (j*numInputChannels)] * scale;
    }

    // update indexing for buffering
    for (int i=0; i < numPCs; i++)
        pc[i].sync_levels();

    // buffer input
    for (int i=0; i < numPCs; i++)
    {
        //printf("-------------------PC: %u\n",i);
        pc[i].input(pc[i].input_mix);
    }

    // start computation threads -------------------------
    if (start_sync_depth >= first_thread_level)
    {
        pthread_mutex_lock(&worker_mutex[start_sync_depth]);

        thread_counter[start_sync_depth] = 0;
        for (int l = first_thread_level; l <= start_sync_depth; l++)
            for (int j = 0; j < threads_in_level[l]; j++)
                worker_state[l][j] = 1;


        pthread_cond_broadcast(&worker_cond[start_sync_depth]);
        pthread_mutex_unlock(&worker_mutex[start_sync_depth]);
    }

    for (int j = threads_in_level[0]; j < total_level0_work_units; j++)
    {
        WorkList *curr = workList[0][j];
        while(curr != NULL)
        {
            curr->obj->runLevel(curr->arg);
            curr = curr->next;
        }
    }


    // end computation threads ------------------------------------
    if (end_sync_depth >= first_thread_level)
    {
        if( thread_counter[end_sync_depth] != sync_target[end_sync_depth]) //check if all 0's to sync depth
        {
            pthread_mutex_lock(&main_mutex[end_sync_depth]);
            while( thread_counter[end_sync_depth] != sync_target[end_sync_depth])
            {
                pthread_cond_wait(&main_cond[end_sync_depth],&main_mutex[end_sync_depth]);
            }
            pthread_mutex_unlock(&main_mutex[end_sync_depth]);
        }
        //reset thread_counter to zero
        int ret = __sync_sub_and_fetch(&thread_counter[end_sync_depth],sync_target[end_sync_depth]);
        if (ret != 0)
            fprintf(stderr,"atomic reset error!\n");

    }


    // copy output -------------------------------------------------
    for (int i=0; i < numPCs; i++)
        pc[i].output(pc[i].output_mix);
    

    memset(output, 0, numOutputChannels*frameCount*sizeof(float));
    for (int i=0; i < numPCs; i++)
    {
        const int chan = outputChannel[i];

        if (!muteChannel[i])
        {
            for (int j = 0; j < frameCount; j++)
                output[chan+(j*numOutputChannels)] += pc[i].output_mix[j];
        }
    }


    frameNum++;
    if (frameNum == lastFrame && doneWaiter != NULL)
        doneWaiter->changeAndSignal(1);
    return 0;
}

void *PartConvMulti::workerThreadEntry(void *arg)
{
    WorkerThreadData *workerThreadData = (WorkerThreadData*)arg;
    PartConvMulti *pc = (PartConvMulti*)workerThreadData->This;
    const int L = workerThreadData->levelNum;
    const int thr = workerThreadData->threadNum;
    WorkList* const localWorkList = pc->workList[L][thr];
    int* const local_state = &pc->worker_state[L][thr];
    int* const thread_counter = pc->thread_counter;
    int* const sync_target = pc->sync_target;

    const int num_levels = pc->num_levels;

    unsigned thread_level_counter[num_levels];
    unsigned thread_level_size[num_levels];
    for (int l = L; l < num_levels; l++)
    {
        thread_level_size[l] = pc->level_size[l]/pc->level_size[L];
        thread_level_counter[l] = thread_level_size[l] - 1;
    }

    printf("thread started: L %u, thr %u\n",L,thr);
    unsigned sync_depth;

    while(!pc->terminate_flag) 
    {
        //figure out number of levels that need to sync
        //assuming that whenever a larger block needs to sync, all smaller blocks do too
        sync_depth = L;
        for (int l = L+1; l < num_levels; l++)
        {
            thread_level_counter[l]++;
            if (thread_level_counter[l] == thread_level_size[l])
            {
                thread_level_counter[l] = 0;
                sync_depth = l;
            }
        }

        __sync_sub_and_fetch (local_state, 1);  //*local_state = 0
        int ret = __sync_add_and_fetch (&thread_counter[sync_depth], 1);  //thread_counter[L] += 1

        if (ret == sync_target[sync_depth])
        {
            // DO NEED MUTEX:
            // if no mutex, sometimes main thread will miss the signal below and never wake up.
            pthread_mutex_lock(&pc->main_mutex[sync_depth]);
            pthread_cond_signal(&pc->main_cond[sync_depth]);
            pthread_mutex_unlock(&pc->main_mutex[sync_depth]);
        }

        
        pthread_mutex_lock(&pc->worker_mutex[sync_depth]);
        while (*local_state == 0)
        {
            pthread_cond_wait(&pc->worker_cond[sync_depth],&pc->worker_mutex[sync_depth]);
        }
        pthread_mutex_unlock(&pc->worker_mutex[sync_depth]);

        if (*local_state < 0) //terminate signal
        {
            break;
        }



        // do work son! ---------------
        WorkList *curr = localWorkList;
        while (curr != NULL)
        {
            curr->obj->runLevel(curr->arg);
            curr = curr->next;
        }
        // --------------------------



    }




    pthread_exit(NULL);
}

/* PartConvTest methods */

int PartConvMultiRelaxed::run(float* const output, const float* const input, const int frameCount) 
{
    if (frameCount != buffer_size)
    {
        fprintf(stderr,"frameCount (%u) != buffer_size (%u)\n",frameCount,buffer_size);
        exit(-1);
    }

    // which levels need to sync?  -------------------
    int sync_depth = 0;

    for (int l = 0; l < num_levels; l++)
    {
        level_counter[l]++;
        if (level_counter[l] == level_size[l])
        {
            level_counter[l] = 0;
            sync_depth = l;
        }
    }

    // deinterleave input
    for (int i = 0; i < numPCs; i++)
    {
        const int chan = inputChannel[i];
        const float scale = scaleFactor[i];

        for (int j = 0; j < frameCount; j++)
            pc[i].input_mix[j] = input[chan + (j*numInputChannels)] * scale;
    }

    // update indexing for buffering
    for (int i=0; i < numPCs; i++)
        pc[i].sync_levels_relaxed();
    

    // end computation threads ------------------------------------
    if (frameNum > 0)
    {
        if (sync_depth >= first_thread_level)
        {
            if( thread_counter[sync_depth] != sync_target[sync_depth]) //check if all 0's to sync depth
            {
                pthread_mutex_lock(&main_mutex[sync_depth]);
                while( thread_counter[sync_depth] != sync_target[sync_depth])
                {
                    pthread_cond_wait(&main_cond[sync_depth],&main_mutex[sync_depth]);
                }
                pthread_mutex_unlock(&main_mutex[sync_depth]);
            }
            //reset thread_counter to zero
            int ret = __sync_sub_and_fetch(&thread_counter[sync_depth],sync_target[sync_depth]);
            if (ret != 0)
                fprintf(stderr,"atomic reset error!\n");

        }
    }


    // buffer output -------------------------------------------------
    for (int i=0; i < numPCs; i++)
        pc[i].output_relaxed(pc[i].output_mix);


    // buffer input
    for (int i=0; i < numPCs; i++)
    {
        //printf("-------------------PC: %u\n",i);
        pc[i].input(pc[i].input_mix);
    }

    // start computation threads -------------------------
    if (sync_depth >= first_thread_level)
    {
        pthread_mutex_lock(&worker_mutex[sync_depth]);

        thread_counter[sync_depth] = 0;
        for (int l = first_thread_level; l <= sync_depth; l++)
            for (int j = 0; j < threads_in_level[l]; j++)
                worker_state[l][j] = 1;


        pthread_cond_broadcast(&worker_cond[sync_depth]);
        pthread_mutex_unlock(&worker_mutex[sync_depth]);
    }

    for (int j = threads_in_level[0]; j < total_level0_work_units; j++)
    {
        WorkList *curr = workList[0][j];
        while(curr != NULL)
        {
            curr->obj->runLevel(curr->arg);
            curr = curr->next;
        }
    }


    

    // copy output --------------------------------------------
    memset(output, 0, numOutputChannels*frameCount*sizeof(float));
    for (int i=0; i < numPCs; i++)
    {
        const int chan = outputChannel[i];

        if (!muteChannel[i])
        {
            for (int j = 0; j < frameCount; j++)
                output[chan+(j*numOutputChannels)] += pc[i].output_mix[j];
        }
    }


    frameNum++;
    if (frameNum == lastFrame && doneWaiter != NULL)
        doneWaiter->changeAndSignal(1);
    return 0;
}

/* Waiter methods */

Waiter::Waiter(int inVal)
{
    initVal = inVal;
    val = initVal;
    int err;
    err = pthread_cond_init(&cond,NULL);
    if (err)
    {
        fprintf(stderr, "Waiter: pthread_cond_init: error\n");
        exit(-1);
    }
    err = pthread_mutex_init(&mutex,NULL);
    if (err)
    {
        fprintf(stderr, "Waiter: pthread_mutex_init: error\n");
        exit(-1);
    }
}

Waiter::~Waiter()
{
    pthread_cond_destroy(&cond);
    pthread_mutex_destroy(&mutex);
}

void Waiter::waitFor(int target)
{
    if (val == target)
    {
        fprintf(stderr, "Waiter::waitFor: target value is same as current value\n");
        return;
    }
    
    pthread_mutex_lock(&mutex);
    while (val != target)
        pthread_cond_wait(&cond,&mutex);
    val = initVal;
    pthread_mutex_unlock(&mutex);
}

void Waiter::changeAndSignal(int newVal)
{
    if (newVal == val)
    {
        fprintf(stderr, "Waiter::changeAndSignal: new value is same as val\n"); 
        return;
    }

    pthread_mutex_lock(&mutex);
    val = newVal;
    pthread_cond_signal(&cond);
    pthread_mutex_unlock(&mutex);
}


/* Timer methods */

/*
Timer::Timer(int inM)
{
    M = inM;
    if (M < 1)
    {
        lockWait = NULL;
        fft = NULL;
        ifft = NULL;
        cmadd = NULL;
        count = NULL;
    }
    else
    {
        lockWait = new double[M+2];
        fft = new double[M];
        ifft = new double[M];
        cmadd = new double[M];
        count = new unsigned[M];
        memset(lockWait, 0, (M+2)*sizeof(double));
        memset(fft, 0, M*sizeof(double));
        memset(ifft, 0, M*sizeof(double));
        memset(cmadd, 0, M*sizeof(double));
        memset(count, 0, M*sizeof(unsigned));
    }
}

Timer::~Timer()
{
    if (M >= 1)
    {
        delete lockWait;
        delete fft;
        delete ifft;
        delete cmadd;
        delete count;
    }
}

void Timer::tally(PartConvFilter* pc)
{
    memcpy( count, pc->bcurr, M*sizeof(unsigned));
}

void Timer::reset()
{
    memset(lockWait, 0, (M+2)*sizeof(double));
    memset(fft, 0, M*sizeof(double));
    memset(ifft, 0, M*sizeof(double));
    memset(cmadd, 0, M*sizeof(double));
    memset(count, 0, M*sizeof(unsigned));
}

void Timer::display(int FS,int* N)
{
    printf("Timing Results\n");
    //printf("Level: lockWait, FFT, IFFT, CMAdd\n");
    printf("%-20s%-30s%-30s%-30s%-30s%-s\n","Level:","FFT","IFFT","CMAdd","Total","Sync");
    for (int L = 0; L < M; L++)
    {
        //printf("%u: %ux (%u): %g (%.2f%%,%.2f%%), %g (%.2f%%,%.2f%%), %g (%.2f%%,%.2f%%), %g (%.2f%%,%.2f%%)\n",
        //        L,N[L]/N[0],count[L],
        //        lockWait[L]/count[L],100*lockWait[L]/count[L]*FS/N[0],100*lockWait[L]/count[L]*FS/N[L],
        //        fft[L]/count[L],100*fft[L]/count[L]*FS/N[0],100*fft[L]/count[L]*FS/N[L],
        //        ifft[L]/count[L],100*ifft[L]/count[L]*FS/N[0],100*ifft[L]/count[L]*FS/N[L],
        //        cmadd[L]/count[L],100*cmadd[L]/count[L]*FS/N[0],100*cmadd[L]/count[L]*FS/N[L]
        //      );
        double sum = fft[L]+ifft[L]+cmadd[L];

        printf("%u: %4ux (%5u): %10f (%6.2f%%,%6.2f%%), %10f (%6.2f%%,%6.2f%%), %10f (%6.2f%%,%6.2f%%), %10f (%6.2f%%,%6.2f%%), %10f (%6.2f%%,%6.2f%%)\n",
                L,N[L]/N[0],count[L],
                fft[L]/count[L],100*fft[L]/count[L]*FS/N[0],100*fft[L]/count[L]*FS/N[L],
                ifft[L]/count[L],100*ifft[L]/count[L]*FS/N[0],100*ifft[L]/count[L]*FS/N[L],
                cmadd[L]/count[L],100*cmadd[L]/count[L]*FS/N[0],100*cmadd[L]/count[L]*FS/N[L],
                sum/count[L],100*sum/count[L]*FS/N[0],100*sum/count[L]*FS/N[L],
                lockWait[L]/count[L],100*lockWait[L]/count[L]*FS/N[0],100*lockWait[L]/count[L]*FS/N[L]
              );
    }

    printf("input buffering: %g (%.2f%%)\n",lockWait[M]/count[0],100*lockWait[M]/count[0]*FS/N[0]);
    printf("output buffering: %g (%.2f%%)\n",lockWait[M+1]/count[0],100*lockWait[M+1]/count[0]*FS/N[0]);
}


int start_timer(Timer* timer, TimerType type, int ind)
{
    if (timer != NULL)
    {
        double *t;
        switch (type)
        {
            case kLockWait:
                t = timer->lockWait;
                break;
            case kFFT:
                t = timer->fft;
                break;
            case kIFFT:
                t = timer->ifft;
                break;
            case kCMAdd:
                t = timer->cmadd;
                break;
            default:
                return 1;
        }
        t[ind] -= get_time();
        return 0;
    }
    else
        return 1;
}

int stop_timer(Timer* timer, TimerType type, int ind)
{
    if (timer != NULL)
    {
        double *t;
        switch (type)
        {
            case kLockWait:
                t = timer->lockWait;
                break;
            case kFFT:
                t = timer->fft;
                break;
            case kIFFT:
                t = timer->ifft;
                break;
            case kCMAdd:
                t = timer->cmadd;
                break;
            default:
                return 1;
        }
        t[ind] += get_time();
        return 0;
    }
    else
        return 1;
}
*/
