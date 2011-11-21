#ifndef _ALSA_H
#define _ALSA_H

#include <alsa/asoundlib.h>
#include "partconvMulti.h"

#ifdef __cplusplus
extern "C"
{
#endif

typedef struct {
  const char *pdevice; // playback device
  const char *cdevice; // capture device
  unsigned int rate;
  unsigned int channels;
  snd_pcm_format_t format;
  snd_pcm_sframes_t buffer_size;
  snd_pcm_uframes_t period_size;
  snd_output_t *output;
  snd_pcm_t *chandle;
  snd_pcm_t *phandle;
  float *audio_in;
  float *audio_out;
} alsa_dev_t;

typedef struct  {
  alsa_dev_t *dev;
  PartConvMulti *pc;
} pc_data_t;

void Float32ToNativeInt32( const float *src, int *dst, unsigned int numToConvert );
void NativeInt32ToFloat32( const int *src, float *dst, unsigned int numToConvert );

int  alsa_set_hwparams(alsa_dev_t *dev, snd_pcm_t *handle, snd_pcm_hw_params_t *params, snd_pcm_access_t access);
int  alsa_set_swparams(alsa_dev_t *dev, snd_pcm_t *handle, snd_pcm_sw_params_t *swparams);
int  alsa_xrun_recovery(snd_pcm_t *handle, int err);
int  alsa_async_direct_loop(alsa_dev_t *dev, void *ptr, void (*callback)(snd_async_handler_t *));
int  alsa_init(alsa_dev_t *dev, void *ptr, void (*callback)(snd_async_handler_t *));
void alsa_fini(alsa_dev_t *dev);
#ifdef __cplusplus
}
#endif
#endif
