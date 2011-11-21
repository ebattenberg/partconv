#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<errno.h>

#include "portaudio.h"
#include "pa_mac_core.h"

#include "partconvMulti.h"

// #define WAVIO //read one input channel from wav file
//#define OUTPUT_NAME "./output.wav"
#define OUTPUT_NAME NULL
#define INPUT_NAME "../audio/input.wav"

#define IMPULSE_NAME "../reverbs/impulse.wav"
#define CONFIG_FILE "pc.config"

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

int outputUnderflowCount = 0;

#ifdef _PORTAUDIO_
int Callback( const void *input,
        void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* paTimeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData);

PaStream *setupAudioStream(PartConvMulti*pc);

#endif

#ifdef _USE_ALSA_
#include "alsa.h"
#include <values.h>

#define AUDIO_SAMPLE_RATE (44100)
#define AUDIO_CHANNELS (2)
#define AUDIO_FRAME_SIZE (64)
#define AUDIO_BUFSIZE_FRAMES (4)

static const char *ALSA_pdevice = "hw:0,0"; /* playback device */
static const char *ALSA_cdevice = "hw:0,0"; /* capture device */

void ALSA_callback(snd_async_handler_t *ahandler);
alsa_dev_t *setupALSA(PartConvMulti*pc);
#endif

int main(int argc, char** argv) 
{
    int fs= 44100;
    int verbosity = 2;

    //PartConvMulti pc(verbosity);
    PartConvMultiRelaxed pc(verbosity);
    int max_threads_per_level = 2;
    int max_level0_threads = 1;
    pc.setup(CONFIG_FILE,max_threads_per_level,max_level0_threads);

#ifdef WAVIO
    IOData io;
    pc.io = io.setup(&pc, INPUT_NAME, OUTPUT_NAME);
    if (pc.io == NULL){
        printf("setup IOData: failed\n");
        exit(-1);
    }
#endif

    int runTime = 0;
    if (argc > 1)
        runTime = atoi(argv[1]);

    if (runTime > 0)
    {
        pc.doneWaiter = new Waiter(0);
        pc.lastFrame = (runTime*fs)/pc.buffer_size;
    }


#if 1
#ifdef _PORTAUDIO_
    printf("Using PortAudio for audio I/O\n");
    PaStream *stream = setupAudioStream(&pc);
    //pc.stream = stream;  
    Pa_StartStream(stream);
    printf("\nrunning stream...\n");
    if (runTime == 0)
    {
        while(1)
        {
            printf("enter to exit: \n");
            if (getc(stdin) == '\n')
                break;
        }
    }
    else
    {
        printf("Running for %u sec (%u frames)\n",runTime,(unsigned)pc.lastFrame);
        pc.doneWaiter->waitFor(1);
    }
    printf("\nstopping stream\n");
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();   
#endif
#endif

#ifdef _USE_ALSA_
    printf("Using ALSA for audio I/O\n");
    pc_data_t apd;

    alsa_dev_t *alsa_dev = setupALSA(&pc);
    assert(alsa_dev != NULL);

    apd.dev = alsa_dev;
    apd.pc = &pc;
    printf("\nrunning stream...\n");
    alsa_init(alsa_dev, &apd, &ALSA_callback);
    printf("\nstopping stream\n");
    alsa_fini(alsa_dev);
#endif

    printf("number of output underflows detected: %d\n", outputUnderflowCount);

#ifdef WAVIO
    io.cleanup();
#endif 
    pc.cleanup();
    if (runTime > 0)
        delete pc.doneWaiter;

    return 0;
}

#ifdef _PORTAUDIO_
int Callback( const void *input,
        void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* paTimeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData)
{
    if ((statusFlags & paOutputUnderflow))
        outputUnderflowCount++;

    PartConvMulti* pc = (PartConvMulti*)userData;
    float* const out = (float* const)output;
    const float* const in = (const float* const)input;

#ifdef WAVIO
    IOData *io = pc->io;
    float *input_mix = pc->input_mix;
    float *audiofile;

    memcpy(input_mix,in,pc->numInputChannels*frameCount*sizeof(float));

    if (io->infile)
    { //mix mic input with audio file
        audiofile = io->getInput();
        for (int i = 0; i < io->block_size; i++)
        {
            //input_mix[pc->numInputChannels*i] += audiofile[i];
            input_mix[pc->numInputChannels*i] = audiofile[i];
        }
    }

    pc->run(out,pc->input_mix,frameCount);

    if(io->infile || io->outfile)
    {
        memset(pc->output_mix,0,pc->numOutputChannels*frameCount*sizeof(float));
        for (int i = 0; i < io->block_size; i++)
            pc->output_mix[i] = out[pc->numOutputChannels*i];
        io->run(pc->output_mix);
    }
#else

    pc->run(out,in,frameCount);

#endif

    return paContinue;
}

PaStream *setupAudioStream(PartConvMulti *pc)
{

    PaStream *stream;
    PaStreamParameters outputParameters,inputParameters;

    Pa_Initialize();

#ifdef __APPLE__
    //outputParameters.device = Pa_GetDefaultOutputDevice();
    outputParameters.device = 6; //Aggregate Device

#else
    outputParameters.device = 1;
    //outputParameters.device = 5; //jack
#endif
    if(outputParameters.device == paNoDevice)
    {
        printf("\nno output device available\n");
        Pa_Terminate();
        exit(-1);
    }

    outputParameters.channelCount = pc->numOutputChannels;
    outputParameters.sampleFormat = paFloat32;
    //outputParameters.suggestedLatency = Pa_GetDeviceInfo( outputParameters.device )->defaultLowOutputLatency;
    //outputParameters.suggestedLatency = (64.0f / 44100.0f);
    outputParameters.suggestedLatency = ((float)(pc->buffer_size) / 44100.0f);


#ifdef __APPLE__
    //inputParameters.device = Pa_GetDefaultInputDevice();
    inputParameters.device = 6; //Aggregate Device
#else
    inputParameters.device = 1;
    //inputParameters.device = 5; //jack
#endif
    if(inputParameters.device == paNoDevice)
    {
        printf("\nno input device available\n");
        Pa_Terminate();
        exit(-1);
    }

    inputParameters.channelCount = pc->numInputChannels;
    inputParameters.sampleFormat = paFloat32;
    inputParameters.suggestedLatency = Pa_GetDeviceInfo( inputParameters.device )->defaultLowInputLatency;
    //inputParameters.suggestedLatency = (64.0f / 44100.0f);
    inputParameters.suggestedLatency = ((float)(pc->buffer_size) / 44100.0f);

    PaStreamFlags streamFlags;
#ifdef __APPLE__
    PaMacCoreStreamInfo coreInfo;

    /*This setting is tuned for pro audio apps. It allows SR conversion on input
      and output, but it tries to set the appropriate SR on the device.*/
    //#define paMacCorePro                         (0x01)
    
    /*This is a setting to minimize CPU usage, even if that means interrupting the device. */
    //#define paMacCoreMinimizeCPU                 (0x0101)
    unsigned long flags = (0x0101);
    PaMacCore_SetupStreamInfo(&coreInfo, flags);
    inputParameters.hostApiSpecificStreamInfo = &coreInfo;
    outputParameters.hostApiSpecificStreamInfo = &coreInfo;

    streamFlags = paNoFlag;

#else
    streamFlags = paNoFlag;
#endif

    PaError error = Pa_OpenStream(
            &stream,
            &inputParameters,
            &outputParameters,
            pc->FS,
            pc->buffer_size,
            streamFlags,
            Callback,
            (void*)pc // user data in callback
            );

    if (error) {
        printf("\nError opening stream, error code = %i\n",error);
        printf("%s\n", Pa_GetErrorText(error));
        Pa_Terminate();
        exit(-1);
    }

    const PaStreamInfo* stream_info = Pa_GetStreamInfo(stream);

#ifndef __APPLE__
    PaAlsa_EnableRealtimeScheduling(stream, true);
#endif

    printf("\nOutput Stream:\n");
    printf("sample rate: %g \n",stream_info->sampleRate);
    printf("device: %u\n",outputParameters.device);
    printf("channels: %u\n",outputParameters.channelCount);
    printf("suggested latency: %g ms\n", outputParameters.suggestedLatency*1000);
    printf("stream latency: %g ms\n", stream_info->outputLatency*1000);

    printf("\nInput Stream:\n");
    printf("sample rate: %g \n",stream_info->sampleRate);
    printf("device: %u\n",inputParameters.device);
    printf("channels: %u\n",inputParameters.channelCount);
    printf("suggested latency: %g ms\n", inputParameters.suggestedLatency*1000);
    printf("stream latency: %g ms\n", stream_info->inputLatency*1000);

    return stream;
}
#endif

#ifdef _USE_ALSA_
static int ALSA_callback_first = 1;

void ALSA_callback(snd_async_handler_t *ahandler)
{
  pc_data_t *data = (pc_data_t *) snd_async_handler_get_callback_private(ahandler);
  alsa_dev_t *dev = data->dev;
  PartConvMulti *pc = data->pc;

  snd_pcm_t *phandle = snd_async_handler_get_pcm(ahandler);
  snd_pcm_t *chandle = dev->chandle; 
  snd_pcm_uframes_t period_size = dev->period_size;
  int channels = dev->channels;

  const snd_pcm_channel_area_t *my_areas;
  snd_pcm_uframes_t offset, frames, size;
  snd_pcm_sframes_t avail, commitres;
  snd_pcm_state_t state;
  int err;
  int first = 0;

  int *sbuf, *dbuf;

  int xrun = 0;
  if (ALSA_callback_first)
  {
    memset(dev->audio_in, 0, AUDIO_FRAME_SIZE*AUDIO_CHANNELS*sizeof(float));
    memset(dev->audio_out, 0, AUDIO_FRAME_SIZE*AUDIO_CHANNELS*sizeof(float));
    ALSA_callback_first = 0;
  }
  else
  {
#ifdef WAVIO
    const unsigned frameCount = AUDIO_FRAME_SIZE;
    assert(pc->numInputChannels == pc->numOutputChannels);

    IOData *io = pc->io;
    float *input_mix = pc->input_mix;
    float *audiofile;

    memcpy(input_mix,dev->audio_in,pc->numInputChannels*frameCount*sizeof(float));

    if (io->infile)
    { //mix mic input with audio file
        audiofile = io->getInput();
        for (int i = 0; i < io->block_size; i++)
        {
            //input_mix[pc->numInputChannels*i] += audiofile[i];
            input_mix[pc->numInputChannels*i] = audiofile[i];
        }
    }

    pc->run(dev->audio_out,pc->input_mix,frameCount);
    //memcpy(dev->audio_out,pc->input_mix,frameCount*pc->numOutputChannels*sizeof(float));

    if(io->infile || io->outfile)
    {
        memset(pc->output_mix,0,pc->numOutputChannels*frameCount*sizeof(float));
        for (int i = 0; i < io->block_size; i++)
            pc->output_mix[i] = dev->audio_out[pc->numOutputChannels*i];
        io->run(pc->output_mix);
    }
#else
    pc->run(dev->audio_out, dev->audio_in, AUDIO_FRAME_SIZE);
#endif
  }

  // write playback data
  do {
	state = snd_pcm_state(phandle);
	if (state == SND_PCM_STATE_XRUN) {
		printf("xrun_recovery(phandle, -EPIPE)\n");
		err = alsa_xrun_recovery(phandle, -EPIPE);
		if (err < 0) {
			printf("XRUN recovery failed: %s\n", snd_strerror(err));
			exit(-1);
		}
		first = 1;
		xrun = 1;
	} else if (state == SND_PCM_STATE_SUSPENDED) {
		printf("xrun_recovery(phandle, -ESTRPIPE)\n");
		err = alsa_xrun_recovery(phandle, -ESTRPIPE);
		if (err < 0) {
			printf("SUSPEND recovery failed: %s\n", snd_strerror(err));
			exit(-1);
		}
	}

    avail = snd_pcm_avail_update(phandle);
    if (avail < 0) {
      err = alsa_xrun_recovery(phandle, avail);
      if (err < 0) {
	      printf("avail update failed: %s\n", snd_strerror(err));
	      exit(EXIT_FAILURE);
      }
      first = 1;
      xrun = 1;
      continue;
    }
    if (avail < period_size) {
	    if (first) {
		    first = 0;
		    err = snd_pcm_start(phandle);
		    if (err < 0) {
			    printf("Start error: %s\n", snd_strerror(err));
			    exit(-1);
		    }
	    } else {
		    break;
	    }
	    continue;
    }
    size = period_size;
    while (size > 0) {
      frames = size;
      err = snd_pcm_mmap_begin(phandle, &my_areas, &offset, &frames);
      if (err < 0) {
	      if ((err = alsa_xrun_recovery(phandle, err)) < 0) {
		      printf("MMAP begin avail error: %s\n", snd_strerror(err));
		      exit(-1);
	      }
	      xrun = 1;
	      first = 1;
      }
      
      dbuf = (int *)(((unsigned char *)my_areas[0].addr) + (my_areas[0].first / 8));
      dbuf += offset * channels;

      Float32ToNativeInt32(dev->audio_out, dbuf, period_size*pc->numOutputChannels);

      commitres = snd_pcm_mmap_commit(phandle, offset, frames);
      if (commitres < 0 || (snd_pcm_uframes_t)commitres != frames) {
	      if ((err = alsa_xrun_recovery(phandle, commitres >= 0 ? -EPIPE : commitres)) < 0) {
		      printf("MMAP commit error: %s\n", snd_strerror(err));
		      exit(-1);
	      }
	      xrun = 1;
      }
      size -= frames;
    }
  } while (1);
  
  // read capture data
  do {
	state = snd_pcm_state(chandle);
	if (state == SND_PCM_STATE_XRUN) {
		printf("xrun_recovery(chandle, -EPIPE)\n");
		err = alsa_xrun_recovery(chandle, -EPIPE);
		if (err < 0) {
			printf("XRUN recovery failed: %s\n", snd_strerror(err));
			exit(-1);
		}
		xrun = 1;
	} else if (state == SND_PCM_STATE_SUSPENDED) {
		printf("xrun_recovery(chandle, -ESTRPIPE)\n");
		err = alsa_xrun_recovery(chandle, -ESTRPIPE);
		if (err < 0) {
			printf("SUSPEND recovery failed: %s\n", snd_strerror(err));
			exit(-1);
		}
	}

    avail = snd_pcm_avail_update(chandle);
    if (avail < 0) {
      err = alsa_xrun_recovery(chandle, avail);
      if (err < 0) {
	      printf("avail update failed: %s\n", snd_strerror(err));
	      exit(-1);
      }
      xrun = 1;
      continue;
    }
    if (avail < period_size) {
      break;
    }
    size = period_size;
    while (size > 0) {
      frames = size;
      err = snd_pcm_mmap_begin(chandle, &my_areas, &offset, &frames);
      if (err < 0) {
	      if ((err = alsa_xrun_recovery(chandle, err)) < 0) {
		      printf("MMAP begin avail error: %s\n", snd_strerror(err));
		      exit(-1);
	      }
	      xrun = 1;
      }

      sbuf = (int *)(((unsigned char *)my_areas[0].addr) + (my_areas[0].first / 8));
      sbuf += offset * channels;
    
      NativeInt32ToFloat32((const int *)sbuf, dev->audio_in, period_size*pc->numInputChannels);  

      commitres = snd_pcm_mmap_commit(chandle, offset, frames);
      if (commitres < 0 || (snd_pcm_uframes_t)commitres != frames) {
	      if ((err = alsa_xrun_recovery(chandle, commitres >= 0 ? -EPIPE : commitres)) < 0) {
		      printf("MMAP commit error: %s\n", snd_strerror(err));
		      exit(-1);
	      }
	      xrun = 1;
      }
      size -= frames;
    }
  } while (1);

    if (xrun)
        outputUnderflowCount++;
}

alsa_dev_t *setupALSA(PartConvMulti *pc)
{
    alsa_dev_t *alsa_dev = (alsa_dev_t *) fftwf_malloc(sizeof(alsa_dev_t));
    assert(alsa_dev != NULL);

    assert(pc->numInputChannels == AUDIO_CHANNELS);
    assert(pc->numOutputChannels == AUDIO_CHANNELS);

    // initialize alsa_dev structure
        alsa_dev->pdevice = ALSA_pdevice;
        alsa_dev->cdevice = ALSA_cdevice;
        alsa_dev->rate = AUDIO_SAMPLE_RATE;
        alsa_dev->channels = AUDIO_CHANNELS;
        alsa_dev->format = SND_PCM_FORMAT_S32_LE;
        alsa_dev->buffer_size = AUDIO_FRAME_SIZE * AUDIO_BUFSIZE_FRAMES;
        alsa_dev->period_size = AUDIO_FRAME_SIZE;
        alsa_dev->output = NULL;
        alsa_dev->audio_in  = (float *) fftwf_malloc(pc->numInputChannels*AUDIO_FRAME_SIZE*sizeof(float));
        alsa_dev->audio_out = (float *) fftwf_malloc(pc->numOutputChannels*AUDIO_FRAME_SIZE*sizeof(float));

        return alsa_dev;
    }

#endif

