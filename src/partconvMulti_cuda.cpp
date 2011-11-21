#ifdef __APPLE__
#include <mach/mach_init.h>
#include <mach/thread_policy.h>
#include <mach/thread_act.h>
#include <mach/mach_time.h>  
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h> //get_nprocs()
#endif

#include <errno.h>

#include "partconvMulti.h"
#include "runlevel.h"

#define BUFF_SECS 1
#define MAX_INPUT_SECS 500
#define CACHE_SIZE (3*1024*1024) //bytes



//#define _DEBUG_
//#define MEASURE_COMPUTE_TIME
//#define _TIME_LEVELS_

#ifdef _CUDA_
#include "cuda.h"
#include "cutil.h"
#endif

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))
#define max(X, Y)  ((X) > (Y) ? (X) : (Y))

void* safe_malloc(size_t size);
void* safe_calloc(size_t count, size_t size);
int checkState(int *state, int start_level, int sync_depth, int check_for);
void AtomicSet(unsigned *ptr, unsigned new_value);
void printState(int *state, int L);
#ifdef __APPLE__
int get_bus_speed();
#endif


void AtomicSet(unsigned *ptr, unsigned new_value)
{
        while (true)
        {
                unsigned old_value = *ptr;
                if (__sync_bool_compare_and_swap(ptr, old_value, new_value)) 
                    return;
        }
}

#ifdef _SNDFILE_
vec readImpulseResponse(const char* filename, int verbosity)
{ 
    sndFileData data;
    data.sndFile = sf_open(filename,SFM_READ,&data.sfInfo);

    if (!data.sndFile) {
        fprintf(stderr,"readImpulseResponse() ERROR: Can't open impulse file: %s\n",filename);
        sf_perror(NULL);
        exit(-1);
    }

    const int numFrames = data.sfInfo.frames;

    const int numChannels = data.sfInfo.channels;

    float* out_array = (float*)safe_malloc(numFrames*sizeof(float));

    if (numChannels > 1){ 
        //remove extra channels
        float* temp = new float[numChannels*numFrames];
        sf_readf_float(data.sndFile,temp,numFrames);

        for(int i=0;i<numFrames;i++)
            out_array[i] = temp[numChannels*i];
        delete[] temp;
    }
    else
        sf_readf_float(data.sndFile,out_array,numFrames);

    sf_close(data.sndFile);
    vec vector;
    vector.size = (unsigned)numFrames;
    vector.data = out_array;

    //FS = data.sfInfo.samplerate;

    if (verbosity > 0)
    {
        printf("\nImpulse response file: %s\n",filename);
        printf("channels: %u (only used one)\n",numChannels);
        printf("frames: %u\n",numFrames);
        printf("format: 0x%x\n",data.sfInfo.format);
        printf("sample rate: %u\n",data.sfInfo.samplerate);
        printf("\n");
    }



    return vector;
}
#endif

int PartConv::setup(
        int M_in, int* N_in, int* num_parts_in, 
        vec *inImpulseResponse, bool inCreateThreads,
        double* inBenching, Timer* inTimer)
{
    // for setup benching
    //timer = new Timer(M_in);
    createThreads = inCreateThreads;
    timer = inTimer;
    double *benchData = inBenching;
    M = M_in;
    

    N = new int[M];
    num_parts = new int[M];
    nfft = new int[M];
    cfft = new int[M];

    memcpy(N,N_in,sizeof(int)*M);
    memcpy(num_parts,num_parts_in,sizeof(int)*M);

    // block size at each level
    for (int i = 0; i < M; i++)
    {
        nfft[i] = 2*N[i];
        cfft[i] = nfft[i]/2 + 1;

        if (N[i] != nextpow2(N[i]))
        {
            fprintf(stderr,"PartConv ERROR: block size of level %u must be a power of 2\n",i);
            fprintf(stderr,"N[%u] = %u\n",i,N[i]);
            delete[] N;
            delete[] num_parts;
            delete[] nfft;
            delete[] cfft;
            return 1;
        }
    }
    for (int i = 1; i < M; i++)
    {
        if (N[i] <= N[i-1])
        {
            fprintf(stderr,"PartConv ERROR: N[%u]=%u <= N[%u]=%u\n",i, N[i], i-1, N[i-1]);
            delete[] N;
            delete[] num_parts;
            delete[] nfft;
            delete[] cfft;
            return 1;
        }
    }


    bcurr = new unsigned[M];
    ind_N0_in_level = new int[M];
    num_N0_in_level = new int[M];

    for (int i = 0; i < M; i++)
    {
        num_N0_in_level[i] = N[i]/N[0];
    }
    for (int i = 0; i < M; i++)
    {
        bcurr[i] = 0;                               //incremented at end of ::runLevel
        ind_N0_in_level[i] = num_N0_in_level[i]-1;  //incremented at beg of ::run
    }

    vec impulse_response;
    matrix h[M];
    if (inImpulseResponse == NULL)
    { //if we're benching, create impulse response of appropriate length
        int length = 0;
        for (int L = 0; L < M; L++)
        {
            length += N[L]*num_parts[L];
        }
        impulse_response = create_vec(length);
        for (int i = 0; i < impulse_response.size; i++)
            impulse_response.data[i] = 1;
    }
    else
    {   
        //impulse_response = readImpulseResponse(impulsename);
        impulse_response = create_vec(inImpulseResponse->size);
        memcpy(impulse_response.data, inImpulseResponse->data, sizeof(float)*impulse_response.size);
    }

    // compute norm for normalizing impulse response
    //float h_norm;
    //h_norm = 0;
    //for (int i = 0; i < impulse_response.size; i++)
    //    h_norm += impulse_response.data[i]*impulse_response.data[i];
    //h_norm = sqrt(h_norm);

    int temp_int = 2*N[0];
    int delay[M];
    for (int i = 0; i < M-1; i++)
    {
        delay[i] = temp_int/N[0];
        temp_int += N[i]*num_parts[i];
        if (temp_int < 2*N[i+1])
        {
            fprintf(stderr,"PartConv ERROR: sufficient delay not allowed for block size of level %u\n",i+2);
            free_vec(&impulse_response);
            delete[] bcurr;
            delete[] ind_N0_in_level;
            delete[] num_N0_in_level;
            delete[] N;
            delete[] num_parts;
            delete[] nfft;
            delete[] cfft;
            return 1;
        }
    }
    delay[M-1] = temp_int/N[0];

    num_parts[M-1] = ceil((float)(impulse_response.size - (temp_int - 2*N[0]))/N[M-1]);  
    if (impulse_response.size - (temp_int - 2*N[0]) < 0)
    {
        fprintf(stderr,"PartConv ERROR: last partitioning level (%u) not required\n",M);
        free_vec(&impulse_response);
        delete[] bcurr;
        delete[] ind_N0_in_level;
        delete[] num_N0_in_level;
        delete[] N;
        delete[] num_parts;
        delete[] nfft;
        delete[] cfft;
        return 1;
    }
    temp_int += N[M-1]*num_parts[M-1];

    vec h_full = zero_pad(impulse_response,temp_int,0);
    int impulse_response_size = impulse_response.size;
    free_vec(&impulse_response);

    // partition impulse response into levels
    vec h_part[M];
    h_part[0].size = N[0]*num_parts[0];
    h_part[0].data = h_full.data;
    for (int i = 1; i < M; i++)
    {
        h_part[i].size = N[i]*num_parts[i];
        h_part[i].data = h_part[i-1].data + h_part[i-1].size;
    }

    // partion levels into blocks
    for (int i = 0; i < M; i++)
    {
        h[i] = splitArray(h_part[i],N[i],nfft[i],N[i]);
    }

    free_vec(&h_full);  //also frees memory used by h_part[]
    for (int i = 1; i < M; i++)
    {
        h_part[i].size = 0;
        h_part[i].data = NULL;
    }

    //Cmatrix H[M];
    H = new Cmatrix[M];

    for (int i = 0; i < M; i++)
        H[i] = create_Cmatrix(num_parts[i],cfft[i]);

    // do the fft on the impluse partitions...
    for (int L = 0; L < M; L++)
    {
        //float scale = nfft[L]*h_norm*2;
        float scale = nfft[L];
        for (int i = 0; i < num_parts[L]; i++)
        {
            fftwf_plan h_fwd = 
                fftwf_plan_dft_r2c_1d(nfft[L], h[L].data[i] ,H[L].Cdata[i] ,FFTW_ESTIMATE);
            fftwf_execute(h_fwd);
            fftwf_destroy_plan(h_fwd);
            // scaling by nfft*(norm of original h)
            for (int j = 0; j < cfft[L]; j++) 
            {
                H[L].Cdata[i][j][0] /= scale;
                H[L].Cdata[i][j][1] /= scale;
            }
        }
    }

    for (int i = 0; i < M; i++)
        free_matrix(&h[i]);

    Xbuf = new DelayBuffer*[M];
    for (int i = 0; i < M; i++)
        Xbuf[i] = new DelayBuffer(N[0],N[i]/N[0],delay[i]);

    Ybuf = new DoubleBuffer*[M];
    for (int i = 0; i < M; i++)
        Ybuf[i] = new DoubleBuffer(N[i]);

    Xfft = new Cmatrix[M];      //holds ffts of input
    inbuffer = new vec[M];      //holds input block
    fftbuffer = new Cvec[M];    //holds fft of input block
    outbuffer = new vec[M];     //holds output block
    Yfft = new Cvec[M];         //holds sum of an FDL
    for (int i = 0; i < M; i++)
    {
        Xfft[i] = create_Cmatrix(num_parts[i],cfft[i]);
        inbuffer[i] = create_vec(nfft[i]); 	  
        fftbuffer[i] = create_Cvec(cfft[i]);  
        outbuffer[i] = create_vec(nfft[i]); 	  
        Yfft[i] = create_Cvec(cfft[i]);
    }


    if (FILE* wisdomfile = fopen("wisdom.wis","r"))
    {
        fftwf_import_wisdom_from_file(wisdomfile);
        fclose(wisdomfile);
    }
    //fftwf_plan bck[M];
    bck = new fftwf_plan[M];
    //fftwf_plan fwd[M]; 
    fwd = new fftwf_plan[M];
    for (int i = 0; i < M; i++)
    {
        bck[i] = fftwf_plan_dft_c2r_1d(nfft[i],Yfft[i].Cdata,outbuffer[i].data,FFTW_PATIENT);
        fwd[i] = fftwf_plan_dft_r2c_1d(nfft[i],inbuffer[i].data,fftbuffer[i].Cdata,FFTW_PATIENT);
    }
    if (FILE* wisdomfile = fopen("wisdom.wis","w"))
    {
        fftwf_export_wisdom_to_file(wisdomfile);
        fclose(wisdomfile);
    }

#ifdef _CUDA_
    //allocate GPU structures
    Xfft_d = (Cvec*)safe_malloc(sizeof(Cvec)*M);
    H_d = (Cvec*)safe_malloc(sizeof(Cvec)*M);
    Yfft_d = (Cvec*)safe_malloc(sizeof(Cvec)*M);
    fftbuffer_d = (Cvec*)safe_malloc(sizeof(Cvec)*M);
    outbuffer_d = (vec*)safe_malloc(sizeof(vec)*M);
    inbuffer_d = (vec*)safe_malloc(sizeof(vec)*M);


    for (int L = 0; L < M; L++)
    {
        Xfft_d[L].size = Xfft[L].rows * Xfft[L].cols;
        H_d[L].size= H[L].rows * H[L].cols;

        Yfft_d[L] = Yfft[L];
        fftbuffer_d[L] = fftbuffer[L];
        outbuffer_d[L] = outbuffer[L];
        inbuffer_d[L] = inbuffer[L];
    }

    // CUFFT plans
    fwd_d = (cufftHandle*)safe_malloc(sizeof(cufftHandle)*M);
    bck_d = (cufftHandle*)safe_malloc(sizeof(cufftHandle)*M);

    // GPU use bools
    useGPU = (int*)safe_malloc(sizeof(int)*M);
#endif

    output_mix = new float[N[0]];
    input_mix = new float[N[0]];
    
    // benching variables
    t = new double[M];
    for(int i = 0; i < M; i++)
        t[i] = 0;

    if (timer != NULL || benchData != NULL)
    {
        //find WCET for each level
        const int pollute_size = CACHE_SIZE/sizeof(int);
        int *pollute_cache = new int[pollute_size];
        const int TRIALS = 100;
        const int WARMS = 5;

        double percent_cold_sum = 0;
        double percent_warm_sum = 0;
        double percent_mean_sum = 0;

        // measure mean performance
        for (int L = 0; L < M; L++)
        {
            double warm = 0;
            double cold = 0;
            double mean = 0;
            double gpu = 0;
            int trial;
#ifdef _CUDA_
            setupGPU(L);
#endif

            for (trial = 0; trial < TRIALS; trial++)
            {
                //pollute the cache
                for (int i = 0; i < pollute_size; i++)
                {   
                    pollute_cache[i] = i;
                    pollute_cache[i]++;
                }

                for (int i = 0; i < WARMS; i++)
                {
                    double t0;
#ifdef _CUDA_
                    t0 = get_time_cuda();
                    runLevelGPU(L);
                    t0 = get_time_cuda() - t0;
                    gpu += t0;
#endif
                    t0 = get_time();
                    runLevel(L);
                    t0 = get_time() - t0;

                    bcurr[L]++;
                    if(i == 0)
                        cold += t0;
                    if(i == WARMS-1)
                        warm += t0;
                    mean += t0;
                }
                if(cold > 0.05)
                {
                    trial++;
                    break;
                }
            }
            mean /= (trial*WARMS);
            gpu /= (trial*WARMS);
            cold /= trial;
            warm /= trial;
            double percent_cold = 100*cold/((double)N[L]/FS);
            double percent_warm = 100*warm/((double)N[L]/FS);
            double percent_mean = 100*mean/((double)N[L]/FS);
            double percent_gpu = 100*gpu/((double)N[L]/FS);

            if (verbosity > 0)
            {
                printf("L:%u cold:%f (%.2f%%), warm:%f (%.2f%%), mean:%f (%.2f%%), gpu:%f (%.2f%%)\n",L,cold,percent_cold,warm,percent_warm,mean,percent_mean,gpu,percent_gpu);
            }
            if (benchData != NULL)
            {
                benchData[2*L] = percent_mean;
                benchData[2*L+1] = percent_cold;
            }

            percent_cold_sum += percent_cold;
            percent_warm_sum += percent_warm;
            percent_mean_sum += percent_mean;

            t[L] = warm; //store cache timing value for use in scheduling

#ifdef _CUDA_
            cleanupGPU(L);
            if (gpu < mean)
            {
                useGPU[L] = 1;
                for (int l = 0; l < L; l++)
                    useGPU[l] = 0;
            }
            else
                useGPU[L] = 0;
#endif

        }

        if (verbosity > 0)
            printf("percentage of total period: cold (%.2f%%), warm (%.2f%%), mean (%.2f%%)\n\n",percent_cold_sum,percent_warm_sum,percent_mean_sum);
        if (benchData != NULL)
        {
            benchData[2*M] = percent_mean_sum;
            benchData[2*M+1] = percent_cold_sum;
        }

        if (timer != NULL)
        {
            printf("\nMEAN PERFORMANCE, WARMS: %u\n",WARMS);
            timer->tally(this);
            if (verbosity > 0)
                timer->display(FS,N);
            timer->reset();
        }
        reset();

        if (timer != NULL)
        {
            // measure warm performance
            for (int L = 0; L < M; L++)
            {
                double warm = 0;
                int trial;

                for (trial = 0; trial < TRIALS; trial++)
                {

                    double t0;
                    t0 = get_time();
                    runLevel(L);
                    t0 = get_time() - t0;

                    bcurr[L]++;

                    warm += t0;
                    if(warm > 0.05*WARMS)
                    {
                        trial++;
                        break;
                    }
                }
                warm /= trial;
                t[L] = warm; //store cache timing value for use in scheduling

            }

            printf("\nWARM PERFORMANCE\n");
            timer->tally(this);
            if (verbosity > 0)
                timer->display(FS,N);
            timer->reset();
            reset();

            // measure cold performance
            for (int L = 0; L < M; L++)
            {
                double cold = 0;
                int trial;

                for (trial = 0; trial < TRIALS; trial++)
                {
                    //pollute the cache
                    for (int i = 0; i < pollute_size; i++)
                    {   
                        pollute_cache[i] = i;
                        pollute_cache[i]++;
                    }

                    double t0;
                    t0 = get_time();
                    runLevel(L);
                    t0 = get_time() - t0;

                    bcurr[L]++;
                    cold += t0;

                    if(cold > 0.05*WARMS)
                    {
                        trial++;
                        break;
                    }
                }
                cold /= trial;
		
            }

            printf("\nCOLD PERFORMANCE\n");
            timer->tally(this);
            if (verbosity > 0)
                timer->display(FS,N);
            timer->reset();
            reset();
        }

        delete[] pollute_cache;
    }
    else
    {
        for (int L = 0; L < M; L++)
        {
            t[L] = 0;
        }
    }
    


    // threading stuff ------------------------------
    if (createThreads)
    {
        first_thread_level = 1;
        terminate_flag = 0;
        pthread_mutex_init(&levels_mutex,NULL);

        // setup pc thread priority (for running higher levels at lower priority)
        pcth = new pthread_t[M];
        pthread_attr_t thread_attr[M];
        worker_cond = new pthread_cond_t[M];
        main_cond = new pthread_cond_t[M];
        pcth_state = new int[M](); //init 0
        pcl = new PartConvLevel[M];         // struct for passing 'this' and L to a thread
        for (int L = first_thread_level; L < M; L++)
        {
            struct sched_param thread_param;
            pthread_attr_init(&thread_attr[L]);
            pthread_attr_setdetachstate(&thread_attr[L], PTHREAD_CREATE_JOINABLE);
            pthread_attr_setschedpolicy(&thread_attr[L], SCHED_FIFO);

            pthread_attr_getschedparam(&thread_attr[L], &thread_param);
#ifdef __APPLE__
            thread_param.sched_priority = sched_get_priority_max(SCHED_FIFO) - L + 16;
#else
            thread_param.sched_priority = 80 - L;
#endif
            pthread_attr_setschedparam(&thread_attr[L], &thread_param);
            pthread_attr_setinheritsched(&thread_attr[L], PTHREAD_EXPLICIT_SCHED);

            pthread_cond_init(&worker_cond[L],NULL);
            pthread_cond_init(&main_cond[L],NULL);

            pcl[L].This = this;
            pcl[L].L = L;

            pcth_state[L] = 1;
        }

        for (int L = first_thread_level; L < M; L++)
        {
            // create a ptread for each level
            int err = pthread_create(&pcth[L], &thread_attr[L], runLevelThreadEntry, (void*)&pcl[L]);
            pthread_attr_destroy(&thread_attr[L]);
            if ( err != 0) 
            {
                printf("pthread_create error: %d\n",err); 
                if (err == EPERM)
                    printf("EPERM\n");
                exit(-1);
            }

#ifdef __APPLE__disabled
            int bus_speed = get_bus_speed();
            thread_time_constraint_policy_data_t time_policy;
            time_policy.period = (uint32_t)bus_speed*(.93985*N[L]/(float)FS);
            //int comp_max = min(5E5,bus_speed*((double).93985*N[0]/FS/M));
            const int comp_max = time_policy.period * 0.9; 
            const int comp_min = 5E4;
            time_policy.computation = (uint32_t)min(comp_max,max(comp_min,(t[L] * bus_speed))); //5E4/1064E6=0.05ms
            time_policy.constraint = (uint32_t)time_policy.period * 0.9;  // would like to leave 10% cushion
            time_policy.preemptible = 1;

            err = thread_policy_set(
                    pthread_mach_thread_np(pcth[L]),
                    THREAD_TIME_CONSTRAINT_POLICY,
                    (thread_policy_t)&time_policy,
                    THREAD_TIME_CONSTRAINT_POLICY_COUNT);
            if (err != KERN_SUCCESS) 
            {
                printf("mach set policy error: %d\n",err); 
                printf("not able to set time constraint policy for level: %u\n",L);
                printf("%u mach cycles: %u %u %u %u %u\n",
                        L,
                        time_policy.period,
                        time_policy.computation,
                        time_policy.constraint,
                        time_policy.preemptible,
                        get_bus_speed()
                      );
                printf("%u mach time: %g %g %g %u %u\n",
                        L,
                        time_policy.period/(float)bus_speed,
                        time_policy.computation/(float)bus_speed,
                        time_policy.constraint/(float)bus_speed,
                        time_policy.preemptible,
                        get_bus_speed()
                      );
            }

#ifdef _DEBUG_
            // check policy params

            mach_msg_type_number_t count = THREAD_TIME_CONSTRAINT_POLICY_COUNT;
            boolean_t get_default = 0;
            err = thread_policy_get(
                    pthread_mach_thread_np(pcth[L]),
                    THREAD_TIME_CONSTRAINT_POLICY,
                    (thread_policy_t)&time_policy,
                    &count,
                    &get_default);
            if ( err != KERN_SUCCESS) 
            {
                printf("mach get policy error: %d (L=%u)\n",err,L); 
                //exit(-1);
            }
            else
            {
                printf("%u mach time: %u %u %u %u %u %u\n",
                        L,
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
                    pthread_mach_thread_np(pcth[L]),
                    THREAD_PRECEDENCE_POLICY,
                    (thread_policy_t)&prec_policy,
                    &count,
                    &get_default);
            if ( err != KERN_SUCCESS) 
            {
                printf("mach get precedence error: %d (L=%u)\n",err,L); 
                //exit(-1);
            }
            else
            {
                printf("%u mach prec: %d %d\n", L, prec_policy.importance, get_default);
            }
#endif
#endif
        }


        int end_sync_depth = M-1;
        pthread_mutex_lock(&levels_mutex);
        while( !checkState(pcth_state, first_thread_level, end_sync_depth, 0)) //check if all 0's to level M-1
            pthread_cond_wait(&main_cond[end_sync_depth],&levels_mutex);
        pthread_mutex_unlock(&levels_mutex);
    }

    // print specs of filter
    if (verbosity > 0)
    {
        printf("\nPartitioning Specs:\n");
        printf("impulse response: %.2f sec\n",(double)impulse_response_size/FS);
        printf("partitioning: ");
        for (int i = 0; i < M-1; i++)
            printf("%ux%u, ", N[i],num_parts[i]);
        printf("%ux%u\n",N[M-1],num_parts[M-1]);
        printf("filter latency: %g ms\n",(float)1000*N[0]/FS);

    }
    
    return 0;
}

int PartConv::cleanup(void)
{

    if (createThreads)
    {
        pthread_mutex_lock(&levels_mutex);
        AtomicSet(&terminate_flag,1); //terminate_flag = 1;

        for (int L = first_thread_level; L < M; L++)
            pcth_state[L] = -1; //terminate signal

        for (int L = first_thread_level; L < M; L++)
            pthread_cond_broadcast(&worker_cond[L]);

        pthread_mutex_unlock(&levels_mutex);

        for (int L = first_thread_level; L < M; L++)
            pthread_join(pcth[L],NULL);
    }

    // tally number of blocks timed
    if (timer != NULL)
    {
        timer->tally(this);
        timer->display(FS,N);
    }

    // free stuff
    for (int i = 0; i < M; i++)
    {
        fftwf_destroy_plan(bck[i]);
        fftwf_destroy_plan(fwd[i]);

        free_vec(&inbuffer[i]);
        free_vec(&outbuffer[i]);
        free_Cvec(&Yfft[i]);
        free_Cvec(&fftbuffer[i]);

        free_Cmatrix(&Xfft[i]);
        free_Cmatrix(&H[i]);


        delete Xbuf[i];
        delete Ybuf[i];
    }

    if (createThreads)
    {
        for (int i = first_thread_level; i < M; i++)
        {
            pthread_cond_destroy(&worker_cond[i]);
            pthread_cond_destroy(&main_cond[i]);
        }
        pthread_mutex_destroy(&levels_mutex);
        delete[] pcth;
        delete[] worker_cond;
        delete[] main_cond;
        delete[] pcth_state;
        delete[] pcl;
    }


    delete[] N;
    delete[] num_parts;
    delete[] nfft;
    delete[] cfft;

    delete[] Yfft;
    delete[] Xfft;
    delete[] inbuffer;
    delete[] outbuffer;
    delete[] fftbuffer;
    delete[] H;
    delete[] fwd;
    delete[] bck;

    delete[] Xbuf;
    delete[] Ybuf;

    delete[] output_mix;
    delete[] input_mix;


    
    
    delete[] bcurr;
    delete[] ind_N0_in_level;
    delete[] num_N0_in_level;


    delete[] t;

#ifdef _CUDA_
    //free GPU structures
    free(Xfft_d);
    free(H_d);
    free(Yfft_d);
    free(fftbuffer_d);
    free(outbuffer_d);
    free(inbuffer_d);

    free(fwd_d);
    free(bck_d);

    free(useGPU);
#endif

    if (verbosity > 0)
        printf("\ncleaned up PartConv struct\n");

    return 0;
}

void PartConv::reset()
{ //reset all internal buffers

    if (timer != NULL)
        timer->reset();

    memset(bcurr,0,M*sizeof(unsigned));
    memset(output_mix,0,N[0]*sizeof(float));
    memset(input_mix,0,N[0]*sizeof(float));

    for (int L = 0; L < M; L++)
    {
        ind_N0_in_level[L] = num_N0_in_level[L] - 1;
        reset_vec(inbuffer[L]);
        reset_Cmatrix(Xfft[L]);
        Xbuf[L]->reset();
        Ybuf[L]->reset();
    }
}

int PartConv::run(float* const output, const float* const input) // (float* const input)
{
    // which levels need to sync?  -------------------
    start_sync_depth = 0;
    end_sync_depth = 0;

    for (int L = 0; L < M; L++)
    {
        ind_N0_in_level[L]++;
        if (ind_N0_in_level[L] == num_N0_in_level[L])
        {
            ind_N0_in_level[L] = 0;
            start_sync_depth = L;
        }
        else if (ind_N0_in_level[L] == num_N0_in_level[L]-1)
            end_sync_depth = L;
    }
    
    //printf("%u: start=%u, end=%u\n",bcurr[0],start_sync_depth,end_sync_depth);
    
    // input section ------------------------------
    start_timer(timer,kLockWait,M);
    for (int L = 0; L < M; L++)
    {
        Xbuf[L]->write(input);
        if (L <= start_sync_depth)
            Xbuf[L]->prepNextRead();
    }
    stop_timer(timer,kLockWait,M);


    
    // start computation threads -------------------------
    if (start_sync_depth > 0)
    {
        //unsigned sync_mask = (1 << (start_sync_depth+1)) - 2;
        pthread_mutex_lock(&levels_mutex);
        //__sync_or_and_fetch(&runningBits, sync_mask);
        //for (int L = 1; L <= start_sync_depth; L++)
        //{
        //    if (__sync_add_and_fetch (&pcth_state[L], 1) != 1)
        //    {
        //        fprintf(stderr,"L=%u, pcth_state[L] != 1\n",L);
        //        exit(1);
        //    }
        //}


        for (int L = 1; L <= start_sync_depth; L++)
            pcth_state[L] = 1;

        pthread_cond_broadcast(&worker_cond[start_sync_depth]);
        pthread_mutex_unlock(&levels_mutex);
    }


    //run L0 FDL in this thread
    runLevel(0); 
    




    // end computation threads ------------------------------------
    if (end_sync_depth > 0)
    {
        if( !checkState(pcth_state, first_thread_level, end_sync_depth, 0)) //check if all 0's to sync depth
        {
            pthread_mutex_lock(&levels_mutex);
            while( !checkState(pcth_state, first_thread_level, end_sync_depth, 0)) //check if all 0's to sync depth
                pthread_cond_wait(&main_cond[end_sync_depth],&levels_mutex);
            pthread_mutex_unlock(&levels_mutex);
        }
    }


    
    // output section -------------------------------------
    start_timer(timer,kLockWait,M+1);

    memcpy(output, outbuffer[0].data, N[0]*sizeof(float));
    for (int L = 1; L < M; L++)
    {
        if (L <= end_sync_depth)
            Ybuf[L]->swap(); //swap read/write buffers

        int readBlock = ind_N0_in_level[L] + 1;
        if (readBlock == num_N0_in_level[L])
            readBlock = 0;
        const float* const ptr = Ybuf[L]->getReadBuffer() + N[0]*readBlock;
        for (int i = 0; i < N[0]; i++)
            output[i] += ptr[i];
    }

    stop_timer(timer,kLockWait,M+1);

#ifdef _DEBUG_
    if (bcurr[0] == 1)
    {
        int err;
        boolean_t get_default = 0;
        mach_msg_type_number_t count = THREAD_PRECEDENCE_POLICY_COUNT;
        thread_precedence_policy_data_t prec_policy;
        err = thread_policy_get(
                mach_thread_self(),
                THREAD_PRECEDENCE_POLICY,
                (thread_policy_t)&prec_policy,
                &count,
                &get_default);
        if ( err != KERN_SUCCESS) 
        {
            printf("mach get precedence error: %d callback\n",err); 
            //exit(-1);
        }
        else
        {
            printf("callback mach prec: %d %d\n", prec_policy.importance, get_default);
        }
    }
#endif

    return 0;
}

double PartConv::bench(const int L, const int TRIALS, const int POLLUTE_EVERY, const float MAX_TIME, double *benchData)
{


    //find WCET for each level
    int *pollute_cache = NULL;
    const int pollute_size = CACHE_SIZE/sizeof(int);

    if (POLLUTE_EVERY > 0)
    {
        pollute_cache = new int[pollute_size];
        for (int i = 0; i < pollute_size; i++)
            pollute_cache[i] = i;
    }


    // measure mean performance
    double mean = 0;
    int trial;

    int cnt = 0;
    for (trial = 0; trial < TRIALS; trial++)
    {
        
        if (POLLUTE_EVERY > 0)
        {
            if (cnt++ == POLLUTE_EVERY)
            {
                cnt = 0;
                //pollute the cache
                for (int i = 0; i < pollute_size; i++)
                    pollute_cache[i]++;
            }
        }

        mean -= get_time();
        runLevel(L);
        mean += get_time();

        bcurr[L]++;

        if(mean > MAX_TIME)
        {
            trial++;
            break;
        }
    }

    mean /= trial;

    if (benchData != NULL)
    {
        int value = 100*mean/((double)N[L]/FS);
        benchData[2*L] = value;
        benchData[2*L+1] = value;
    }

    if (POLLUTE_EVERY > 0)
        delete[] pollute_cache;
    reset();
    return mean;



    reset();
    return mean;


}

void PartConv::sync_levels()
{
    // which levels need to sync?  -------------------
    start_sync_depth = 0;
    end_sync_depth = 0;

    for (int L = 0; L < M; L++)
    {
        ind_N0_in_level[L]++;
        if (ind_N0_in_level[L] == num_N0_in_level[L])
        {
            ind_N0_in_level[L] = 0;
            start_sync_depth = L;
        }
        else if (ind_N0_in_level[L] == num_N0_in_level[L]-1)
            end_sync_depth = L;
    }
}

void PartConv::input(const float* const input)
{
    for (int L = 0; L < M; L++)
    {
        //printf("\tL%u:\n",L);
        Xbuf[L]->write(input);
        if (L <= start_sync_depth)
            Xbuf[L]->prepNextRead();
    }
}

void PartConv::start()
{
    if (!createThreads)
        fprintf(stderr,"PartConv: no threads to start\n");
    // start computation threads -------------------------
    if (start_sync_depth > 0)
    {
        pthread_mutex_lock(&levels_mutex);

        for (int L = 1; L <= start_sync_depth; L++)
            pcth_state[L] = 1;

        pthread_cond_broadcast(&worker_cond[start_sync_depth]);
        pthread_mutex_unlock(&levels_mutex);
    }

    //run L0 FDL in this thread
    runLevel(0); 
}

void PartConv::end()
{
    if (!createThreads)
        fprintf(stderr,"PartConv: no threads to end\n");

    if (end_sync_depth > 0)
    {
        if( !checkState(pcth_state, first_thread_level, end_sync_depth, 0)) //check if all 0's to sync depth
        {
            pthread_mutex_lock(&levels_mutex);
            while( !checkState(pcth_state, first_thread_level, end_sync_depth, 0)) //check if all 0's to sync depth
                pthread_cond_wait(&main_cond[end_sync_depth],&levels_mutex);
            pthread_mutex_unlock(&levels_mutex);
        }
    }
}

void PartConv::output(float* const output)
{
    memcpy(output, outbuffer[0].data, N[0]*sizeof(float));
    for (int L = 1; L < M; L++)
    {
        if (L <= end_sync_depth)
            Ybuf[L]->swap(); //swap read/write buffers

        int readBlock = ind_N0_in_level[L] + 1;
        if (readBlock == num_N0_in_level[L])
            readBlock = 0;
        const float* const ptr = Ybuf[L]->getReadBuffer() + N[0]*readBlock;
        for (int i = 0; i < N[0]; i++)
            output[i] += ptr[i];
    }
}

void *PartConv::runLevelThreadEntry(void *arg)
{
    PartConvLevel *pcl = (PartConvLevel*)arg;
    PartConv *pc = (PartConv*)pcl->This;
    const int L = pcl->L;
    const int M = pc->M;
    const int* const N = pc->N;

#ifdef _CUDA_
    pc->setupGPU(L);
#endif

    unsigned NinNL[M];
    unsigned ind_in_level[M];
    for (int l = L; l < M; l++)
    {
        NinNL[l] = N[l]/N[L];    // how many L-blocks fit into an l-block?
        ind_in_level[l] = NinNL[l]-1;  //where is the current L-block within an l-block?
    }

    unsigned sync_depth;
        

    while(!pc->terminate_flag) 
    {
        //figure out number of levels that need to sync
        //assuming that whenever a larger block needs to sync, all smaller blocks do too
        sync_depth = L;
        //next_sync_depth = L;
        for (int l = L+1; l < M; l++)
        {
            ind_in_level[l]++;
            if (ind_in_level[l] == NinNL[l])
            {
                ind_in_level[l] = 0;
                sync_depth = l;
            }
        }

        pthread_mutex_lock(&pc->levels_mutex);
        pc->pcth_state[L] -= 1; //thread finished


        if (checkState(pc->pcth_state, pc->first_thread_level, sync_depth, 0))  //if all 0's to sync depth
            pthread_cond_signal(&pc->main_cond[sync_depth]);
        
        //pthread_mutex_lock(&pc->levels_mutex);
        while (pc->pcth_state[L] == 0)
            pthread_cond_wait(&pc->worker_cond[sync_depth],&pc->levels_mutex);

        if (pc->pcth_state[L] == -1)
        {
            pthread_mutex_unlock(&pc->levels_mutex);
            break;
        }
        else
            pthread_mutex_unlock(&pc->levels_mutex);
        

#ifdef _CUDA_
        //printf("L: %u, useGPU: %u\n",L,pc->useGPU[L]);
        if (pc->useGPU[L])
            pc->runLevelGPU(L);
        else 
            pc->runLevel(L);
#else
        pc->runLevel(L);
#endif
    }

#ifdef _CUDA_
    pc->cleanupGPU(L);
#endif

    pthread_exit(NULL);
}


void* safe_malloc(size_t size)
{
    void* ptr = malloc(size);
    if (ptr == NULL)
    {
        fprintf(stderr,"safe_malloc: NULL returned for chunk of size %u\n", (unsigned)size);
        exit(-1);
    }
    else 
        return ptr;
}

void* safe_calloc(size_t count, size_t size)
{
    void* ptr = calloc(count,size);
    if (ptr == NULL)
    {
        fprintf(stderr,"safe_calloc: NULL returned for chunk of size %u\n", (unsigned)size);
        exit(-1);
    }
    else 
        return ptr;
}
    

#ifdef _SNDFILE_
IOData::IOData()
{
    verbosity = 1;
}

IOData::~IOData()
{
}

int IOData::cleanup()
{
    //close audio files
    if (infile != NULL)
    {
        pthread_mutex_lock(&ith_mutex);
        ith_state = -1;
        pthread_cond_signal(&ith_cond);
        pthread_mutex_unlock(&ith_mutex);
        sf_close(infile->sndFile);
        free_vec(&xbuf);
        free_vec(&xread);
        if (infile->sfInfo.channels > 1)
            free_vec(&xtemp);
        delete infile;

        pthread_mutex_destroy(&ith_mutex);
        pthread_cond_destroy(&ith_cond);
    }
    if (outfile != NULL)
    {
        pthread_mutex_lock(&oth_mutex);
        oth_state = -1;
        pthread_cond_signal(&oth_cond);
        pthread_mutex_unlock(&oth_mutex);

        sf_close(outfile->sndFile);
        free_vec(&ybuf);
        free_vec(&ywrite);
        delete outfile;

        pthread_mutex_destroy(&oth_mutex);
        pthread_cond_destroy(&oth_cond);
    }

    pthread_attr_destroy(&ioth_attr);

    return 0;
}

IOData* IOData::setup(PartConvMulti* pc, const char* infilename, const char* outfilename)
{
    const int N0 = pc->buffer_size;
    const int FS = pc->FS;

    block_size = N0;
    io_ind = 0;

    
    // setup io thread priority 
    struct sched_param ioth_param;
    pthread_attr_init(&ioth_attr);
    pthread_attr_setdetachstate(&ioth_attr, PTHREAD_CREATE_DETACHED);
    pthread_attr_setschedpolicy(&ioth_attr, SCHED_FIFO);
    ioth_param.sched_priority = sched_get_priority_min(SCHED_FIFO)+1;
    pthread_attr_setschedparam(&ioth_attr, &ioth_param);
    pthread_attr_setinheritsched(&ioth_attr, PTHREAD_EXPLICIT_SCHED);

    // prepare input audio file
    if (infilename == NULL)
    {
        infile = NULL;
    }
    else
    {
        infile = new sndFileData;
        
        infile->position = 0;
        infile->sfInfo.format = 0;
        infile->sndFile = sf_open(infilename,SFM_READ, &infile->sfInfo);

        if (!infile->sndFile)
        {
            fprintf(stderr,"IOData: Can't open input audio file %s\n",infilename);
            return NULL;
        }
        if (FS != infile->sfInfo.samplerate)
        {
            fprintf(stderr,"IOData: input file sample rate does not match impulse response sample rate\n");
            return NULL;
        }

        infile->sfInfo.frames = min(infile->sfInfo.frames, MAX_INPUT_SECS*FS);

        int temp_int = max(ceil(BUFF_SECS*FS/(float)N0), 10); // read in BUFF_SEC audio data into xbuf
        xbuf = create_vec(temp_int*N0);
        xread = create_vec(temp_int*N0);
        if (infile->sfInfo.channels > 1)
            xtemp = create_vec(temp_int*N0*infile->sfInfo.channels);
        else
        {
            xtemp.size = 0;
            xtemp.data = NULL;
        }

        num_input_blocks = temp_int;

        readChunk(); //read part of input file into xread
        vec temp_vec;

        // swap xread and xbuf
        temp_vec = xbuf;
        xbuf = xread;
        xread = temp_vec;

        pthread_mutex_init(&ith_mutex,NULL);
        pthread_cond_init(&ith_cond,NULL);

        ith_state = 0;

        int err = pthread_create(&ith, &ioth_attr, readChunkThreadEntry, (void*)this);
        if ( err != 0) 
        {
            printf("pthread_create error: %d\n",err); 
            if (err == EPERM)
                printf("EPERM\n");
            exit(-1);
        }

    }

    // prepare output audio file
    if (outfilename == NULL)
    {
        outfile = NULL;
    }
    else
    {
        outfile = new sndFileData;

        outfile->position = 0;
        outfile->sfInfo.samplerate = FS;
        outfile->sfInfo.channels = 1;

        //outfile->sfInfo.format = (SF_ENDIAN_CPU | SF_FORMAT_AU | SF_FORMAT_FLOAT); 
        outfile->sfInfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;
        outfile->sndFile = sf_open(outfilename,SFM_WRITE, &outfile->sfInfo);

        if (!outfile->sndFile)
        {
            fprintf(stderr,"IOData: Can't open output audio file %s\n",outfilename);
            return NULL;
        }

        int temp_int = max(ceil(BUFF_SECS*FS/(float)N0), 10); // read in INPUT_SEC audio data into xbuf
        ybuf = create_vec(temp_int*N0);
        ywrite = create_vec(temp_int*N0);

        num_output_blocks = temp_int;

        vec temp_vec;
        // swap 
        temp_vec = ybuf;
        ybuf = ywrite;
        ywrite = temp_vec;

        pthread_mutex_init(&oth_mutex,NULL);
        pthread_cond_init(&oth_cond,NULL);

        oth_state= 0;

        int err = pthread_create(&oth, &ioth_attr, writeChunkThreadEntry, (void*)this);
        if ( err != 0) 
        {
            printf("pthread_create error: %d\n",err); 
            if (err == EPERM)
                printf("EPERM\n");
            exit(-1);
        }


    }


    return this;

}

void* IOData::readChunkThreadEntry(void *pThis)
{
    IOData *io = (IOData*)pThis;

    while(1) //loop until ith_state == -1
    {
        pthread_mutex_lock(&io->ith_mutex);
        while(io->ith_state == 0 || io->ith_state == 1)
        {
            pthread_cond_wait(&io->ith_cond,&io->ith_mutex);
        }
        if(io->ith_state == -1)
            break;

        io->readChunk();

        io->ith_state = 1; //tell main thread that computation is done
        pthread_cond_signal(&io->ith_cond);
        pthread_mutex_unlock(&io->ith_mutex);

    }

    pthread_exit(NULL);

}

int IOData::readChunk(void) 
{ // thread to read audio from file into xbuf
    
    const int samples_needed = xread.size;
    const int stride = infile->sfInfo.channels;

    int thisRead = infile->sfInfo.frames - infile->position;

    if (thisRead > 0)
    {
        //sf_seek(infile->sndFile, infile->position, SEEK_SET);

        if (samples_needed > thisRead)
        { // need more samples to fill buffer than are remaining in file
            if (stride > 1)
            {
                sf_readf_float(infile->sndFile, xtemp.data, thisRead);
                for (int i = 0; i < thisRead; i++)
                    xread.data[i] = xtemp.data[stride*i];
            }
            else
                sf_readf_float(infile->sndFile, xread.data, thisRead);

            // zero remaining
            memset(xread.data+thisRead,0, sizeof(float)*(samples_needed-thisRead));

            printf("reached end of input file\n");
        } 
        else 
        {
            thisRead = samples_needed;
            if (stride > 1)
            {
                sf_readf_float(infile->sndFile, xtemp.data, thisRead);
                for (int i = 0; i < thisRead; i++)
                    xread.data[i] = xtemp.data[stride*i];
            }
            else
                sf_readf_float(infile->sndFile, xread.data, thisRead);
        }


    }
    else
    { // no data remaining in file, fill buffer with zeros
        thisRead = 0;
        memset(xread.data,0, sizeof(float)*samples_needed);
    }

    infile->position += thisRead;

    if (verbosity > 1)
        printf("Read %u samples from disk\n",thisRead);
    
    return 0;
}

void* IOData::writeChunkThreadEntry(void *pThis)
{
    IOData *io = (IOData*)pThis;

    while(1) //loop until oth_state == -1
    {
        pthread_mutex_lock(&io->oth_mutex);
        while(io->oth_state == 0 || io->oth_state == 1)
        {
            pthread_cond_wait(&io->oth_cond,&io->oth_mutex);
        }
        if(io->oth_state == -1)
            break;

        io->writeChunk();

        io->oth_state = 1; //tell main thread that computation is done
        pthread_cond_signal(&io->oth_cond);
        pthread_mutex_unlock(&io->oth_mutex);
    }
    pthread_exit(NULL);

}

int IOData::writeChunk(void) 
{ // thread to read audio from file into xbuf
    
    const int thisWrite = ybuf.size;

    sf_writef_float(outfile->sndFile, ywrite.data, thisWrite);

    outfile->position += thisWrite;

    if (verbosity > 1)
        printf("Wrote %u samples to disk\n",thisWrite);
    
    return 0;
}

float* IOData::getInput()
{
    float* input;
    if (infile != NULL)
        input = xbuf.data+(block_size* (io_ind%num_input_blocks)); 
    else 
        input = NULL;
    return input;
    
}

int IOData::run(const float* const output)
{


    if (infile != NULL)
    {
        // position within disk read buffer
        int input_ind = io_ind % num_input_blocks;

        if (input_ind+1 == num_input_blocks)
        {
            pthread_mutex_lock(&ith_mutex);
            while (ith_state == 2) //thread still running but mutex acquired
            {
                pthread_cond_wait(&ith_cond,&ith_mutex);
                fprintf(stderr,"WARNING IOData::run() input read started late\n");
            }
            if (ith_state == 1) //join thread
            {
                ith_state = 0;
            }

            vec temp_vec = xbuf;
            xbuf = xread;
            xread = temp_vec;
        }
        if (ith_state == 0)
        { 
            ith_state = 2;  //2 is "run" signal
            pthread_cond_signal(&ith_cond);
            pthread_mutex_unlock(&ith_mutex);

        }
    }
    if (outfile != NULL)
    {
        // position within disk write buffer
        int output_ind = io_ind % num_output_blocks;

        // write output to disk
        memcpy(ybuf.data+output_ind*block_size, output, block_size*sizeof(float));

        if (output_ind+1 == num_output_blocks)
        {
            pthread_mutex_lock(&oth_mutex);
            while (oth_state == 2) //thread still running but mutex acquired
            {
                pthread_cond_wait(&oth_cond,&oth_mutex);
                fprintf(stderr,"WARNING IOData::run() output write started late\n");
            }
            if (oth_state == 1) //join thread
            {
                oth_state = 0;
            }

            if (oth_state == 0)
            {
                vec temp_vec = ybuf;
                ybuf = ywrite;
                ywrite = temp_vec;

                //start thread
                oth_state = 2;  //2 is "run" signal
                pthread_cond_signal(&oth_cond);
                pthread_mutex_unlock(&oth_mutex);

            }
        }
    }


    io_ind++;

    return io_ind-1;

}
#endif //_SNDFILE_


#ifdef __APPLE__
int get_bus_speed()
{
    int mib[2];
    unsigned int miblen;
    int busspeed;
    int retval;
    size_t len;

    mib[0]=CTL_HW;
    mib[1]=HW_BUS_FREQ;
    miblen=2;
    len=4;
    retval = sysctl(mib, miblen, &busspeed, &len, NULL, 0);

    return busspeed;
}
#endif



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

void Timer::tally(PartConv* pc)
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


DelayBuffer::DelayBuffer(int inFrameSize, int inBlockSize, int delay)
{
    frameSize = inFrameSize;
    framesPerBlock = inBlockSize;
    // already a delay from compute time and initial N[0] buffering
    readFrameDelay = delay - framesPerBlock - 1;

    // round buffer size up to next multiple of block size
    int rem = (readFrameDelay+1) % framesPerBlock;
    bufferSize = readFrameDelay + 1 + (framesPerBlock - rem);
    numBlocks = bufferSize/framesPerBlock + 1; //extra block to prevent read/write overlap 
    bufferSize = numBlocks*framesPerBlock;
    //bufferMem = (float*) safe_calloc( bufferSize*frameSize,sizeof(float));
    bufferMem = new float[bufferSize*frameSize]();
    
    //buffer = (float**) safe_malloc( bufferSize*sizeof(float*));
    buffer = new float*[bufferSize];
    for (int i = 0; i < bufferSize; i++)
        buffer[i] = bufferMem + i*frameSize;

    readBlock = numBlocks-1;
    writeBlock = readFrameDelay/framesPerBlock;
    writeFrame = readFrameDelay % framesPerBlock;

}

DelayBuffer::~DelayBuffer()
{
    delete[] bufferMem;
    delete[] buffer;
}

void DelayBuffer::write(const float* const x)
{
    // write mem at x to current frame
    int frame = writeBlock*framesPerBlock + writeFrame;
    memcpy(buffer[frame],x,frameSize*sizeof(float));
    //printf("wrote to %u.%u of %u.%u\n",writeBlock,writeFrame,numBlocks,framesPerBlock);
    writeFrame++;
    if (writeFrame == framesPerBlock)
    {
        writeFrame = 0;
        writeBlock++;
        if (writeBlock == numBlocks)
            writeBlock = 0;
    }

}

void DelayBuffer::read(float* y)
{
    // copy current block to mem at y
    int frame = readBlock * framesPerBlock;
    if (readBlock == writeBlock)
    {
        printf("ERROR: readBlock (%u) == writeBlock (%u)\n",readBlock,writeBlock);
        printf("frameSize = %u, framesPerBlock = %u, numBlocks = %u\n",frameSize,framesPerBlock,numBlocks);
        exit(-1);
        //printf("readBlock0 = %u, writeBlock0 = %u, writeFrame0 = %u\n", numBlocks-1,readFrameDelay/framesPerBlock,readFrameDelay % framesPerBlock);
        return;
    }
    memcpy(y, buffer[frame], frameSize*framesPerBlock*sizeof(float));
    //readBlock += 1;
    //if (readBlock == numBlocks)
    //    readBlock = 0;

}

void DelayBuffer::reset()
{
    memset(bufferMem,0,bufferSize*sizeof(float));
    readBlock = numBlocks-1;
    writeBlock = readFrameDelay/framesPerBlock;
    writeFrame = readFrameDelay % framesPerBlock;
}

void DelayBuffer::prepNextRead()
{
    readBlock += 1;
    if (readBlock == numBlocks)
        readBlock = 0;
    //printf("prepped read from %u.0 of %u.0\n",readBlock,numBlocks);
}


DoubleBuffer::DoubleBuffer(int inSize)
{
    size = inSize;
    readBuffer = new float[size]();
    writeBuffer = new float[size]();
}

DoubleBuffer::~DoubleBuffer()
{
    delete[] readBuffer;
    delete[] writeBuffer;
}

void DoubleBuffer::reset()
{
    memset(readBuffer,0,size*sizeof(float));
    memset(writeBuffer,0,size*sizeof(float));
}

float* DoubleBuffer::getReadBuffer()
{
    return readBuffer;
}

float* DoubleBuffer::getWriteBuffer()
{
    return writeBuffer;
}

void DoubleBuffer::swap()
{
    float* temp = readBuffer;
    readBuffer = writeBuffer;
    writeBuffer = temp;
}



int checkState(int *state, int start_level, int sync_depth, int check_for)
{
    for (int L = start_level; L <= sync_depth; L++)
    {
        if (state[L] != check_for)
            return 0;
    }
    return 1;
}

void printState(int *state, int L)
{
    printf("state: ");
    for (int i = 0; i < L; i++)
    {
        printf("%u ",state[i]);
    }
    printf("\n");
}

int PartConvMulti::setup( const char* config_file_name, int inThreadsPerLevel, double min_work)
{
    int ret;
    FILE *config_file;
    char ir_filename[256]; // impulse response filename

    config_file = fopen(config_file_name, "r");

    if (config_file == NULL) {
        fprintf(stderr, "Unable to open config file %s\n", config_file_name);
        exit(-1);
    }

    ret = fscanf(config_file, "%d %d %d %d %d\n", &buffer_size, &numInputChannels, &numOutputChannels, &numPCs, &numRepeats);
    if (ret != 5) {
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

    pc = new PartConv[numPCs*numRepeats];
    
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

    for (int i = 0; i < numPCs; i++)
    {

        Timer *timer;
        if (verbosity > 2)
        {
            pc[i].verbosity = 1;
            timer = new Timer(M[i]);
        }
        else
        {
            pc[i].verbosity = 0;
            timer = NULL;
        }


        

#ifdef _SNDFILE_
        bool create_threads = false;
        double *benchData = NULL;
        vec impulse_response = readImpulseResponse(filenames[i]);
        ret = pc[i].setup(M[i],N[i],num_parts[i],&impulse_response,create_threads,benchData,timer);
        free_vec(&impulse_response);
        if (ret) {
            printf("setup PartConv %u: failed\n",i);
            exit(-1);
        }
#else
        fprintf(stderr,"PartConvMulti::setup() libsndfile not enabled\n");
#endif
        delete[] N[i];
        delete[] num_parts[i];
        delete[] filenames[i];

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
                double t = pc[i].bench(L,trials,pollute_every,max_time);
                //double t = 0.01;
                if (verbosity > 1)
                {
                    printf("%2u: %6u x %-4u: %4.2fms (%5.2f%%) [%6.2f%%]\n",j,pc[i].N[L],pc[i].num_parts[L],
                            t*1000, 100*t/((double)pc[i].N[L]/FS), 100*t/((double)pc[i].N[0]/FS));
                }
                exec_time[i][j] = t;
            }
        }

    }
    delete[] N;
    delete[] M;
    delete[] num_parts;
    delete[] filenames;

    // divide up work amongst threads
    // for now we cluster adjacent work units,
    // could get more uniform work division if lifting this constraint
    max_threads_per_level = inThreadsPerLevel;
    threads_in_level = new int[num_levels](); //init 0's
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
                work_target += 1.0;
        }
        work_target /= max_threads_per_level;
#endif

        if (verbosity > 1)
            printf("%u: work_target = %f\n",l,work_target);

        int thr = 0;
        double work_sum = 0;
        for (int indPC = 0; indPC < numPCs; indPC++)
        {
            if (level_map[indPC][l] < 0)
                continue;

            if (work_sum >= work_target)
            {
                work_sum = 0;
                thr++;
                if (thr == max_threads_per_level)
                {
                    fprintf(stderr,"work partitioning error\n");
                    exit(-1);
                }
            }

            // add work unit to linked list
            workList[l][thr] = new WorkList(workList[l][thr]);
#ifdef MEASURE_COMPUTE_TIME
            const double t = exec_time[indPC][l];
#else
            const double t = 1.0;
#endif

            workList[l][thr]->arg = level_map[indPC][l];
            workList[l][thr]->ind = indPC;
            workList[l][thr]->obj = &pc[indPC];
            workList[l][thr]->time = exec_time[indPC][l];
            work_sum += t;
            work_level_sum[l] += exec_time[indPC][l];
            if (verbosity > 1)
                printf("level: %u, thr: %u, indPC: %u, new_work: %f, work_sum: %f\n",l,thr,indPC,t,work_sum);
        }
        threads_in_level[l] = thr+1;
        if (verbosity > 1)
            printf("threads total: %u\n",threads_in_level[l]);


    }


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

        }

    }


    if (verbosity > 0) //print work per level
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
        thread_param.sched_priority = actual_max_priority - i - 1;
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
#endif
        }
        pthread_attr_destroy(&thread_attr[i]);

    }


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
        delete pc[i].timer;
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
{ //reset all internal buffers
    // do we need to reset any other state

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



int PartConvMax::setup(int n, int m, int k, float* impulses, int filterLength, int stride, int* blockSizes, int levels)
{
    int ret;

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


    pc = new PartConv[numPCs*numRepeats];
    
    //numPCs *= numRepeats;

    
    level_counter = new unsigned[num_levels]; 
    level_size = new unsigned[num_levels];
    level_map = NULL;

    for (int i = 0; i < num_levels; i++)
    {
        level_size[i] = blockSizes[i]/blockSizes[0];
        level_counter[i] = level_size[i] - 1;
    }


    vec impulse_response = create_vec(filterLength);
    for (int i = 0; i < numPCs; i++)
    {
        if (stride == 1)
            memcpy(impulse_response.data, &impulses[filterLength*i], sizeof(float)*filterLength);
        else
        {
            for (int l = 0; l < filterLength; l++)
                impulse_response.data[l] = impulses[stride*(filterLength*i + l)];
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

            //vec impulse_response = readImpulseResponse(filenames[i]);
            ret = pc[ind].setup(num_levels,blockSizes,num_parts,&impulse_response);

            if (ret) {
                printf("setup PartConv %u.%u (%u): failed\n",i,j,ind);
                return 1;
            }
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

    free_vec(&impulse_response);
    delete[] num_parts;

    // divide up work amongst threads
    max_threads_per_level = 1;
    threads_in_level = new int[num_levels](); //init 0's
    workList = new WorkList**[num_levels];
    workList[0] = new WorkList*[num_levels*max_threads_per_level](); //init 0's (NULL's)
    for (int i = 1; i < num_levels; i++)
        workList[i] = workList[0] + i*max_threads_per_level;


    double work_level_sum[num_levels];
    for (int l = 0; l < num_levels; l++)
    {
#ifdef MEASURE_COMPUTE_TIME
        const int trials = 1000;
        const int pollute_every = 3;
        const float max_time = 0.1;
        const double t = pc[0].bench(l,trials,pollute_every,max_time);
#else
        const double t = 1.0;
#endif

        work_level_sum[l] = 0;

        double work_target = 0;
        for(int i = 0; i < numPCs; i++)
        {
            work_target += t;
        }
        work_target /= max_threads_per_level;

        int thr = 0;
        double work_sum = 0;
        for (int indPC = 0; indPC < numPCs; indPC++)
        {

            if (work_sum >= work_target)
            {
                work_sum = 0;
                thr++;
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
            work_sum += t;
            work_level_sum[l] += t;
            if (verbosity > 1)
                printf("level: %u, thr: %u, indPC: %u, new_work: %f, work_sum: %f\n",l,thr,indPC,t,work_sum);
        }
        threads_in_level[l] = thr+1;
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
        const int actual_max_priority = 63; //apple low balls the call to sched_get_priority_max(SCHED_FIFO)
        thread_param.sched_priority = actual_max_priority - i - 1;

        pthread_attr_setschedparam(&thread_attr[i], &thread_param);
        pthread_attr_setinheritsched(&thread_attr[i], PTHREAD_EXPLICIT_SCHED);


        //printf("%u: level_time = %f\n",i,level_time*1000);

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


#if defined(__APPLE__) && defined(MEASURE_COMPUTE_TIME)
            const int bus_speed = get_bus_speed();
            const float level_time = (float)level_size[i]*buffer_size/FS;
            double work_sum = 0;
            WorkList *curr = workList[i][j];
            while (curr != NULL)
            {
                work_sum += curr->time;
                curr = curr->next;
            }
            if (verbosity > 0)
                printf("%u.%u: work_sum = %f ms\n",i,j,work_sum*1000);

            thread_time_constraint_policy_data_t time_policy;
            time_policy.period = (uint32_t)bus_speed * .93985 * level_time;
            const uint32_t comp_max = 5E7; // 47ms @ bus_speed = 1064000000
            const uint32_t comp_min = 5E4; // 47us
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
#endif
        }
        pthread_attr_destroy(&thread_attr[i]);

    }


    int end_sync_depth = num_levels-1;
    pthread_mutex_lock(&main_mutex[end_sync_depth]);
    while( thread_counter[end_sync_depth] != sync_target[end_sync_depth])
        pthread_cond_wait(&main_cond[end_sync_depth],&main_mutex[end_sync_depth]);
    pthread_mutex_unlock(&main_mutex[end_sync_depth]);


    printf("Total PartConv instances   : %d\n\n", numPCs);
    printf("buffer_size: %u\n",buffer_size);
    frameNum = 0;
    lastFrame = 0;
    doneWaiter = NULL;



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
