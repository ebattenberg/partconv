#include<stdio.h>
#include<stdlib.h>
#include<assert.h>
#include<errno.h>

#include "portaudio.h"
#include "pa_mac_core.h"

#include "partconv.h"
#include "sndtools.h"

#define IMPULSE_NAME "../reverbs/hamilton_mausoleum.wav"
#define PARTITIONING {64,256,1024,4096,16384,65536}
#define LEVELS 6

#define min(X, Y)  ((X) < (Y) ? (X) : (Y))

int outputUnderflowCount = 0;

int Callback( const void *input,
        void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* paTimeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData);

PaStream *setupAudioStream(PartConvMax*pc);


int main(int argc, char** argv) 
{
    int fs= 44100;
    int verbosity = 1;
    int ret;

    int partitioning[] = PARTITIONING;

    PartConvMax pc(verbosity);

    
    Vector impulses;
    if ( readImpulseResponse(impulses,IMPULSE_NAME,1) )
    { 
        fprintf(stderr, "Could not read impulse responses  from disk\n");
        exit(-1);
    }
    const int stride = 1;
    const int fftwOptLevel = 2;

    ret = pc.setup(fs,2,2,1,&impulses[0],impulses.getSize(), stride, partitioning, LEVELS,fftwOptLevel,"wisdom.wis");
    if (ret)
    {
        fprintf(stderr,"PartConvMax::setup() error\n");
        return 1;
    }



    printf("Using PortAudio for audio I/O\n");
    PaStream *stream = setupAudioStream(&pc);
    Pa_StartStream(stream);
    printf("\nrunning stream...\n");
    while(1)
    {
        printf("enter to exit: \n");
        if (getc(stdin) == '\n')
            break;
    }
    printf("\nstopping stream\n");
    Pa_StopStream(stream);
    Pa_CloseStream(stream);
    Pa_Terminate();   


    printf("number of output underflows detected: %d\n", outputUnderflowCount);

    pc.cleanup();

    return 0;
}

int Callback( const void *input,
        void *output,
        unsigned long frameCount,
        const PaStreamCallbackTimeInfo* paTimeInfo,
        PaStreamCallbackFlags statusFlags,
        void *userData)
{
    if ((statusFlags & paOutputUnderflow))
        outputUnderflowCount++;

    PartConvMax* pc = (PartConvMax*)userData;
    float* const out = (float* const)output;
    const float* const in = (const float* const)input;

    if (frameCount != (unsigned)pc->buffer_size)
    {
        fprintf(stderr,"Callback: frameCount != pc->buffer_size\n");
        return paContinue;
    }


    const int inChannels = pc->numInputChannels;
    const int outChannels = pc->numOutputChannels;
    const int numSamples = pc->buffer_size;

    for (int i = 0; i < inChannels; i++)
    {
        for (int j = 0; j < numSamples; j++)
            pc->inbuffers[i][j] = in[i + j*inChannels];
    }


    pc->run(pc->outbuffers,pc->inbuffers);

    for (int i = 0; i < outChannels; i++)
    {
        for (int j = 0; j < numSamples; j++)
            out[i + j*outChannels] = pc->outbuffers[i][j];
    }



    return paContinue;
}

PaStream *setupAudioStream(PartConvMax *pc)
{
    PaStream *stream;
    PaStreamParameters outputParameters,inputParameters;

    Pa_Initialize();

    //outputParameters.device = Pa_GetDefaultOutputDevice();
    outputParameters.device = 6; //Aggregate Device

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


    //inputParameters.device = Pa_GetDefaultInputDevice();
    inputParameters.device = 6; //Aggregate Device
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

