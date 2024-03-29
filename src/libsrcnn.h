#ifndef __LIBSRCNN_H__
#define __LIBSRCNN_H__

#ifdef LIBSRCNNSTATIC
    #define DLL_PUBLIC
    #define DLL_LOCAL
#else
#if defined _WIN32 || defined __CYGWIN__
  #ifdef BUILDING_DLL
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllexport))
    #else
      #define DLL_PUBLIC __declspec(dllexport)
    #endif
  #else
    #ifdef __GNUC__
      #define DLL_PUBLIC __attribute__ ((dllimport))
    #else
      #define DLL_PUBLIC __declspec(dllimport)
    #endif
  #endif
  #define DLL_LOCAL
#else
  #if __GNUC__ >= 4
    #define DLL_PUBLIC __attribute__ ((visibility ("default")))
    #define DLL_LOCAL  __attribute__ ((visibility ("hidden")))
  #else
    #define DLL_PUBLIC
    #define DLL_LOCAL
  #endif
#endif
#endif /// of LIBSRCNNSTATIC

// libsrcnn version means,  0.1.10.40
#define LIBSRCNN_VERSION    0x00010A28

typedef enum DLL_PUBLIC
{
    SRCNNF_Nearest = 0,
    SRCNNF_Bilinear,
    SRCNNF_Bicubic,
    SRCNNF_Lanczos3,
    SRCNNF_Bspline
}SRCNNFilterType;

void DLL_PUBLIC ConfigureFilterSRCNN( SRCNNFilterType ftype,
                                      bool stepscale  = false );
int  DLL_PUBLIC ProcessSRCNN( const unsigned char* refbuff,
                              unsigned w, unsigned h, unsigned d,
                              float multiply,
                              unsigned char* &outbuff,
                              unsigned &outbuffsz,
                              unsigned char** convbuff,
                              unsigned* convbuffsz);

#endif /// of __SRCNN_H__
