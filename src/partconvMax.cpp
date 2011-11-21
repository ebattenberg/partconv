/* PartConvMax methods */

int PartConvMax::setup(int inSampleRate, int n, int m, int k, float* impulses, int filterLength, int stride, int* blockSizes, int levels, int fftwOptLevel, const char* wisdom)
{
    int ret;
    
    FS = inSampleRate;

    numInputChannels = n;
    numOutputChannels = m;
    if ( (n == m) && (n == k) )  //parallel filtering with distinct impulse responses
    {
        numPCs = k;
        numRepeats = 1;
        inputChannel = new int[numPCs];
        outputChannel = new int[numPCs];

        for (int i = 0; i < numPCs; i++)
        {
            inputChannel[i] = i;
            outputChannel[i] = i;
        }
    }
    else if ( (n == m) && (k == 1) ) //parallel with identical impulse reponse
    {
        
        numPCs = 1;
        numRepeats = n;
        inputChannel = new int[numRepeats];
        outputChannel = new int[numRepeats];

        for (int i = 0; i < numRepeats; i++)
        {
            inputChannel[i] = i;
            outputChannel[i] = i;
        }
    }
    else if ( m == n*k ) //filterbank
    {
        numPCs = k;
        numRepeats = n;
        inputChannel = new int[numPCs*numRepeats];
        outputChannel = new int[numPCs*numRepeats];

        for (int i = 0; i < numPCs; i++)
        {
            for (int j = 0; j < numRepeats; j++)
            {
                inputChannel[i*numRepeats+j] = j;
                outputChannel[i*numRepeats+j] = i*numRepeats+j;
            }
        }
    }
    else
    {
        fprintf(stderr, "PartConvMax::setup() does not support requested channel configuration\n");
        return 1; 
    }



    buffer_size = blockSizes[0];
    num_levels = levels;

    int *num_parts = new int[num_levels]();

    int total_length = 0;
    for (int i = 0; i < num_levels-1; i++)
    {
        num_parts[i] = 2*blockSizes[i+1]/blockSizes[i] - 2;
        total_length += blockSizes[i]*num_parts[i];
        // check if partitioning is already long enough for filter
        if (total_length >= filterLength)
        {
            num_levels = i+1;
            break;
        }
    }
    
    //output_mix = new float[buffer_size*numOutputChannels];
    //input_mix = new float[buffer_size*numInputChannels];
    outbuffers = new float*[numOutputChannels];
    inbuffers = new float*[numInputChannels];
    outbuffers[0] = new float[buffer_size*numOutputChannels];
    inbuffers[0] = new float[buffer_size*numInputChannels];
    for (int i = 1; i < numOutputChannels; i++)
        outbuffers[i] = outbuffers[0] + i*buffer_size;
    for (int i = 1; i < numInputChannels; i++)
        inbuffers[i] = inbuffers[0] + i*buffer_size;


    pc = new PartConvFilter[numPCs*numRepeats];
    
    //numPCs *= numRepeats;

    
    level_counter = new unsigned[num_levels]; 
    level_size = new unsigned[num_levels];
    level_map = NULL;

    for (int i = 0; i < num_levels; i++)
    {
        level_size[i] = blockSizes[i]/blockSizes[0];
        level_counter[i] = level_size[i] - 1;
    }



    if (wisdom != NULL)
    {
        if (FILE* wisdomfile = fopen(wisdom,"r"))
        {
            fftwf_import_wisdom_from_file(wisdomfile);
            fclose(wisdomfile);
        }
    }

    Vector impulseResponse(filterLength);
    for (int i = 0; i < numPCs; i++)
    {
        if (stride == 1)
            impulseResponse.copyFrom(&impulses[filterLength*i]);
        else
        {
            for (int l = 0; l < filterLength; l++)
                impulseResponse[l] = impulses[stride*(filterLength*i + l)];
        }

        for (int j = 0; j < numRepeats; j++)
        {
            const int ind = i*numRepeats+j;
            if (verbosity > 2)
            {
                pc[ind].verbosity = 1;
            }
            else
            {
                pc[ind].verbosity = 0;
            }

            ret = pc[ind].setup(num_levels,blockSizes,num_parts,&impulseResponse,fftwOptLevel);

            if (ret)
            {
                printf("setup PartConvFilter %u.%u (%u): failed\n",i,j,ind);
                return 1;
            }
        }
    }
    if (wisdom != NULL)
    {
        if (FILE* wisdomfile = fopen(wisdom,"w"))
        {
            fftwf_export_wisdom_to_file(wisdomfile);
            fclose(wisdomfile);
        }
    }

    if (verbosity > 0)
    {
        printf("\nPartitioning scheme:\n");
        for(int i = 0; i < num_levels-1; i++)
        {
            printf("%ux%u, ",pc[0].N[i],pc[0].num_parts[i]);
        }
            printf("%ux%u",pc[0].N[num_levels-1],pc[0].num_parts[num_levels-1]);
        printf("\n\n");
    }

    // now treat all PCs the same whether they are repeats or not.
    numPCs *= numRepeats;
    numRepeats = 1;

    delete[] num_parts;

    // divide up work amongst threads
    max_threads_per_level = 1;
    threads_in_level = new int[num_levels](); //init 0's
    workList = new WorkList**[num_levels];
    workList[0] = new WorkList*[num_levels*max_threads_per_level](); //init 0's (NULL's)
    for (int i = 1; i < num_levels; i++)
        workList[i] = workList[0] + i*max_threads_per_level;





    const float callback_time = (float)buffer_size/FS;
    double work_sum[num_levels][max_threads_per_level];
    double work_level_sum[num_levels];

#ifdef MEASURE_COMPUTE_TIME
    bool time_constraint_flag = true;
#else
    bool time_constraint_flag = false;
#endif

    bool time_constraint_policy[num_levels];
    for (int l = 0; l < num_levels; l++)
    {  
        /* benchmark partconv filters until a level exceeds a percentage of callback,
         * then just set time to one and don't allow time constraint policy
         * */

        double t;
        if (time_constraint_flag)
        {
            const int trials = 1000;
            const int pollute_every = 3;
            const float max_time = 0.1;
            t = pc[0].bench(l,trials,pollute_every,max_time);
        }
        else
            t = 1.0;

        double work_target = 0;
        for(int i = 0; i < numPCs; i++)
        {
            work_target += t;
        }
        work_target /= max_threads_per_level;

        work_level_sum[l] = 0;
        int thr = 0;
        work_sum[l][thr] = 0;
        for (int indPC = 0; indPC < numPCs; indPC++)
        {

            if (work_sum[l][thr] >= work_target)
            {
                thr++;
                work_sum[l][thr] = 0;
                if (thr == max_threads_per_level)
                {
                    fprintf(stderr,"work partitioning error\n");
                    exit(-1);
                }
            }

            // add work unit to linked list
            workList[l][thr] = new WorkList(workList[l][thr]);

            workList[l][thr]->arg = l;
            workList[l][thr]->ind = indPC;
            workList[l][thr]->obj = &pc[indPC];
            workList[l][thr]->time = t;
            work_sum[l][thr] += t;
            work_level_sum[l] += t;
            if (verbosity > 1)
                printf("level: %u, thr: %u, indPC: %u, new_work: %f, work_sum: %f\n",
                        l,thr,indPC,t,work_sum[l][thr]);
        }
        threads_in_level[l] = thr+1;
        if (time_constraint_flag && work_level_sum[l] < 0.2*callback_time)
            time_constraint_policy[l] = true;
        else
        {
            time_constraint_flag = false;
            time_constraint_policy[l] = false;
        }

        
        if (verbosity > 1)
            printf("threads total: %u\n",threads_in_level[l]);

    }




    // setup threading stuff ----------------------------------------------

    first_thread_level = 1;
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
        for (int j = first_thread_level; j <= i; j++)
            sync_target[i] += threads_in_level[j];
    }

    for (int i = first_thread_level; i < num_levels; i++)
    {
        pthread_cond_init(&worker_cond[i],NULL);
        pthread_mutex_init(&worker_mutex[i],NULL);
        pthread_cond_init(&main_cond[i],NULL);
        pthread_mutex_init(&main_mutex[i],NULL);
    }


    for (int i = first_thread_level; i < num_levels; i++)
    {
        struct sched_param thread_param;
        pthread_attr_init(&thread_attr[i]);
        pthread_attr_setdetachstate(&thread_attr[i], PTHREAD_CREATE_JOINABLE);
        pthread_attr_setschedpolicy(&thread_attr[i], SCHED_FIFO);

        pthread_attr_getschedparam(&thread_attr[i], &thread_param);

        //thread_param.sched_priority = sched_get_priority_max(SCHED_FIFO) - i + 15;
#ifdef _PORTAUDIO_
        // for some reason, we have to set it explicitly here
        const int actual_max_priority = 63; 
#else
        // when running in Max/MSP, 48 actually maps to 63
        const int actual_max_priority = 48; //apple low balls the call to sched_get_priority_max(SCHED_FIFO)
#endif
        thread_param.sched_priority = actual_max_priority - log2i(level_size[i]) + 1;
        // level-0 always runs in callback thread.  

        pthread_attr_setschedparam(&thread_attr[i], &thread_param);
        pthread_attr_setinheritsched(&thread_attr[i], PTHREAD_EXPLICIT_SCHED);



        for (int j = 0; j < threads_in_level[i]; j++)
        {
            workerThreadData[i][j].This = this;
            workerThreadData[i][j].levelNum = i;
            workerThreadData[i][j].threadNum = j;

            worker_state[i][j] = 1;

            int err = pthread_create(&worker_thread[i][j], &thread_attr[i], workerThreadEntry, (void*)&workerThreadData[i][j]);
            if ( err != 0) 
            {
                printf("pthread_create error: %d\n",err); 
                if (err == EPERM)
                    printf("EPERM\n");
                exit(-1);
            }


#ifdef __APPLE__ 
            if (verbosity > 1)
                printf("level: %u.%u, worksum: %g, callback_time: %g, %u\n",
                        i,j,work_sum[i][j],callback_time,time_constraint_policy[i]);

            if (time_constraint_policy[i])
            {
                thread_time_constraint_policy_data_t time_policy;
                const int bus_speed = get_bus_speed();
                const float level_time = (float)level_size[i]*buffer_size/FS;


                time_policy.period = (uint32_t)bus_speed * .93985 * level_time;
                const uint32_t comp_max = 5E7; // 47ms @ bus_speed = 1064000000
                const uint32_t comp_min = 5E4; // 47us
                time_policy.computation = (uint32_t)(work_sum[i][j] * bus_speed);
                time_policy.computation = max( time_policy.computation, comp_min); //5E4/1064E6=0.05ms
                time_policy.computation = min( time_policy.computation, comp_max); //5E4/1064E6=0.05ms
                time_policy.constraint = (uint32_t)time_policy.period * 0.7;  // would like to leave cushion
                time_policy.preemptible = 1;
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

            printf("\nSETUP PRIORITIES OF %u.%u\n",i,j);
            struct sched_param thread_param;
            int policy;
            pthread_getschedparam(worker_thread[i][j], &policy, &thread_param);
            if (policy == SCHED_FIFO)
                printf("pthread policy: SCHED_FIFO\n");
            else
                printf("pthread policy: not SCHED_FIFO\n");
            printf("priority: %u\n",thread_param.sched_priority);


            thread_time_constraint_policy_data_t time_policy;
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
#endif //_DEBUG_
#endif //__APPLE__
        }
        pthread_attr_destroy(&thread_attr[i]);

    }


    int end_sync_depth = num_levels-1;
    pthread_mutex_lock(&main_mutex[end_sync_depth]);
    while( thread_counter[end_sync_depth] != sync_target[end_sync_depth])
        pthread_cond_wait(&main_cond[end_sync_depth],&main_mutex[end_sync_depth]);
    pthread_mutex_unlock(&main_mutex[end_sync_depth]);


    printf("Total PartConvFilter instances   : %d\n\n", numPCs);
    printf("buffer_size: %u\n",buffer_size);
    frameNum = 0;






    return 0;
}

int PartConvMax::cleanup(void)
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
    delete[] pc;

    //delete[] input_mix;
    //delete[] output_mix;
    delete[] outbuffers[0];
    delete[] inbuffers[0];
    delete[] outbuffers;
    delete[] inbuffers;

    delete[] level_counter; 
    delete[] level_size;


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

int PartConvMax::run(float** const output, float** const input) 
{
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

    // update buffering indexing
    for (int i=0; i < numPCs; i++)
        pc[i].sync_levels();

    // buffer input
    for (int i=0; i < numPCs; i++)
    {
        pc[i].input(input[inputChannel[i]]);
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

    for (int i = 0; i < first_thread_level; i++)
    {
        for (int j = 0; j < threads_in_level[i]; j++)
        {
            WorkList *curr = workList[i][j];
            while(curr != NULL)
            {
                curr->obj->runLevel(curr->arg);
                curr = curr->next;
            }
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


    // copy output
    for (int i=0; i < numPCs; i++)
        pc[i].output(output[outputChannel[i]]);

    frameNum++;

    return 0;
}

void PartConvMax::reset()
{ 
    //reset all internal buffers
    // do we need to reset any other state?

    for (int i = 0; i < numPCs; i++)
        pc[i].reset();
}


void *PartConvMax::workerThreadEntry(void *arg)
{
    WorkerThreadData *workerThreadData = (WorkerThreadData*)arg;
    PartConvMax *pc = (PartConvMax*)workerThreadData->This;
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

int PartConvFilter::runLevel(int L)
{

    int slot;

    memcpy(&inbuffer[L][0], &inbuffer[L][N[L]], sizeof(float)*N[L]);
    Xbuf[L]->read(&inbuffer[L][N[L]]);

    //fft slot
    slot = bcurr[L] % num_parts[L];
    fftwf_execute(fwd[L]); // take fft of inbuffer, save in fftbuffer
    memcpy(Xfft[L][slot], fftbuffer[L](), sizeof(fftwf_complex)*cfft[L]);

    // reset Yfft to zeros
    memset(Yfft[L](), 0, sizeof(fftwf_complex)*cfft[L]);

    // do filtering
    for (int p = 0; p < num_parts[L]; p++) 
    {
        slot = (bcurr[L]-p + num_parts[L]) % num_parts[L];
        const float *Aptr = (const float *)Xfft[L][slot];
        const float *Bptr = (const float *)H[L][p];
        float *Cptr = (float *)Yfft[L]();

        
        __m128 A, B, C, D;
#ifndef _DONT_UNROLL_CMULT_LOOP
        for (int i = 0; i < cfft[L]-1; i+=8)
        {
            A = _mm_load_ps(Aptr);
            B = _mm_load_ps(Bptr);
            C = _mm_load_ps(Cptr);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+4);
            B = _mm_load_ps(Bptr+4);
            C = _mm_load_ps(Cptr+4);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+4, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+8);
            B = _mm_load_ps(Bptr+8);
            C = _mm_load_ps(Cptr+8);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+8, C);
            
            // unroll
            
            A = _mm_load_ps(Aptr+12);
            B = _mm_load_ps(Bptr+12);
            C = _mm_load_ps(Cptr+12);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr+12, C);
            
            Aptr += 16;
            Bptr += 16;
            Cptr += 16;
        }
#else
        for (int i = 0; i < cfft[L]-1; i+=2)
        {
            A = _mm_load_ps(Aptr);
            B = _mm_load_ps(Bptr);
            C = _mm_load_ps(Cptr);
            
            D = _mm_moveldup_ps(A);
            D = _mm_mul_ps(D, B);
            
            A = _mm_movehdup_ps(A);
            B = _mm_shuffle_ps(B, B, 0xB1);
            A = _mm_mul_ps(A, B);
            
            D = _mm_addsub_ps(D, A);
            C = _mm_add_ps(C, D);
            _mm_store_ps(Cptr, C);
            
            Aptr += 4;
            Bptr += 4;
            Cptr += 4;
        }
#endif
        Cptr[0]  += (Aptr[0] * Bptr[0]) - (Aptr[1] * Bptr[1]);
        Cptr[1]  += (Aptr[0] * Bptr[1]) + (Aptr[1] * Bptr[0]); 

        
    }

    // take ifft of FDL
    fftwf_execute(bck[L]); //take ifft of Yfft, save in outbuffer
    
    //copy output into double buffer
    memcpy(Ybuf[L]->getWriteBuffer(),outbuffer[L](), N[L]*sizeof(float));

    bcurr[L]++;

    return 0;

}
