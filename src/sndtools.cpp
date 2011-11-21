#include "sndtools.h"

int readImpulseResponse(Vector & v, const char* filename, int verbosity)
{ 
    sndFileData data;
    data.sndFile = sf_open(filename,SFM_READ,&data.sfInfo);


    if (!data.sndFile) {
        fprintf(stderr,"readImpulseResponse() ERROR: Can't open audio file: %s\n",filename);
        sf_perror(NULL);
        return 1;
    }

    const int numFrames = data.sfInfo.frames;
    const int numChannels = data.sfInfo.channels;

    v.create(numFrames);

    if (numChannels > 1){ 
        //remove extra channels
        float* temp = new float[numChannels*numFrames];
        sf_readf_float(data.sndFile,temp,numFrames);

        for(int i=0;i<numFrames;i++)
            v[i] = temp[numChannels*i];
        delete[] temp;
    }
    else
        sf_readf_float(data.sndFile,&v[0],numFrames);

    sf_close(data.sndFile);


    if (verbosity > 0)
    {
        printf("\nImpulse response file: %s\n",filename);
        printf("channels: %u (only used one)\n",numChannels);
        printf("frames: %u\n",numFrames);
        printf("format: 0x%x\n",data.sfInfo.format);
        printf("sample rate: %u\n",data.sfInfo.samplerate);
        printf("\n");
    }


    return 0;
}

void write_wav(const char* filename, Vector x, double samplerate, int channels) 
{
    // we strip metadata from the input wav files when we save out...
    SNDFILE *outfile;
    SF_INFO sfinfo;

    sfinfo.samplerate = samplerate;
    sfinfo.channels = channels;
    // SF_FORMAT_FLOAT is not what we want for 16 bit wav
    sfinfo.format = SF_FORMAT_WAV | SF_FORMAT_PCM_16;

    if (!(outfile = sf_open(filename,SFM_WRITE, &sfinfo))) {
        printf("Unable to open output wav file: %s\n",filename);
        sf_perror(NULL);
        exit(-1);
    }
    sf_write_float(outfile, x(), x.getSize());
    sf_close(outfile);

    printf("wrote %s\n",filename); 
}

/*
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
*/
