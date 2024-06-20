/*******************************************************************************
** libSRCNN: Library of Super-Resolution with deep Convolutional Neural Networks
** ----------------------------------------------------------------------------
** Current Author : Raphael Kim ( rageworx@gmail.com )
** Previous Author : Wang Shu ( https://github.com/shuwang127/SRCNN_Cpp )
** Referenced to : http://mmlab.ie.cuhk.edu.hk/projects/SRCNN.html
**
** [ Updates ]
**
** - 2023-03-08 -
**     Layer I + II at once, by zvezdochiot@github.com
**
** - 2018-08-08 -
**     First C++ code ( non-OpenCV ) code released.
**     Tested with MinGW-W64 and G++ @ AARCH64 ( nVidia Jetson TX2 )
**
** - 2018-08-09 -
**     Enhanced & Fixed codes to best performance for OpenMP.
**
** - 2019-08-23 -
**     Recursive calling option by factor 2.0 when over scaling.
**
** - 2019-08-26 -
**     Fixed color channels don't use bicubic filter.
**     Now supporting alpha channel.
**
*******************************************************************************/
////////////////////////////////////////////////////////////////////////////////
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>

#ifndef NO_OMP
    #include <omp.h>
#endif

#include "libsrcnn.h"
#include "frawscale.h"
#include "minmax.h"

/* pre-calculated convolutional data */
#include "convdata.h"

#ifdef DEBUG
#include "debugtool.h"
#endif

////////////////////////////////////////////////////////////////////////////////

using namespace std;

////////////////////////////////////////////////////////////////////////////////

namespace libsrcnn {

////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    unsigned       width;
    unsigned       height;
    unsigned       depth;
    unsigned char* buff;
}ImgU8;

typedef struct
{
    unsigned       width;
    unsigned       height;
    unsigned       depth;
    float*         buff;
}ImgF32;

typedef struct
{
    ImgF32      Y;
    ImgF32      Cb;
    ImgF32      Cr;
    bool        uA;
    ImgF32      A;
}ImgYCbCr;

typedef ImgF32  ImgConv1Layers[CONV1_FILTERS];
typedef ImgF32  ImgConv2Layers[CONV2_FILTERS];

////////////////////////////////////////////////////////////////////////////////

static bool             intp_stepscale  = false;
static SRCNNFilterType  intp_filter     = SRCNNF_Bicubic;

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgF32 &src, ImgF32 &dst, \
                    const KernelMat99 kernel, float bias );
void convolution11( ImgConv1Layers &src, ImgYCbCr &dst, \
                    const ConvKernel1 kernel, float bias );
void convolution55( ImgConv2Layers &src, ImgF32 &dst, \
                    const ConvKernel32_55 kernel, float bias );
void Convolution99x11( ImgF32& src, ImgF32* dst, const ConvKernel64_99 kernel99, \
                                                 const ConvKernel1 bias99, \
                                                 const ConvKernel21 kernel11, \
                                                 const ConvKernel2 bias11 );

////////////////////////////////////////////////////////////////////////////////

// some utility functions here ...

inline unsigned uTrim( int64_t a, int64_t b, int64_t c )
{
    int64_t buff[3] = {a, c, b};
    int64_t mpos = (c > a) + (c > b);
    if ( mpos < 0 ) mpos = 0;
    return buff[ mpos ];
}

inline unsigned uTrim32( uint32_t a, uint32_t b, uint32_t c )
{
    uint32_t buff[3] = {a, c, b};
    uint32_t mpos = (c > a) + (c > b);
    if ( mpos < 0 ) mpos = 0;
    return buff[ mpos ];
}

void resetImgU8( ImgU8 &img )
{
    img.width = 0;
    img.height = 0;
    img.depth = 0;

    if ( img.buff != NULL )
    {
        delete[] img.buff;
        img.buff = NULL;
    }
}

void initImgU8( ImgU8 &img, unsigned w, unsigned h, unsigned d )
{
    img.width  = w;
    img.height = h;
    img.depth  = d;

    unsigned imgsz = w * h * d;
    img.buff = new unsigned char[ imgsz ];
}

void resetImgF32( ImgF32 &img )
{
    img.width = 0;
    img.height = 0;
    img.depth = 0;

    if ( img.buff != NULL )
    {
        delete[] img.buff;
        img.buff = NULL;
    }
}

void initImgF32( ImgF32 &img, unsigned w, unsigned h )
{
    img.width = w;
    img.height = h;
    img.depth = 1;

    unsigned buffsz = w * h;
    img.buff = new float[ buffsz ];
}

void initImgConvLayers( ImgF32* img, unsigned w, unsigned h, unsigned count )
{
    if ( img != NULL )
    {
        for( unsigned cnt=0; cnt<count; cnt++ )
        {
            img[cnt].width  = w;
            img[cnt].height = h;
            img[cnt].depth  = 1;

            unsigned buffsz = w * h;
            img[cnt].buff = new float[ buffsz ];
        }
    }
}

void discardConvLayers( ImgF32* img, unsigned count )
{
    if ( img != NULL )
    {
        for( unsigned cnt=0; cnt<count; cnt++ )
        {
            if ( img[cnt].buff != NULL )
            {
                delete[] img[cnt].buff;
                img[cnt].buff = NULL;
            }
        }
    }
}

void discardImgYCbCr( ImgYCbCr &img )
{
    resetImgF32( img.Y );
    resetImgF32( img.Cb );
    resetImgF32( img.Cr );

    if ( img.uA == true )
    {
        resetImgF32( img.A );
    }
}

void initImgYCbCr( ImgYCbCr &img, unsigned w, unsigned h, unsigned d )
{
    initImgF32( img.Y, w, h );
    initImgF32( img.Cb, w, h );
    initImgF32( img.Cr, w, h );

    if ( d == 4 )
    {
        img.uA = true;
        initImgF32( img.A, w, h );
    }
    else
    {
        memset( &img.A, 0, sizeof( ImgF32 ) );
    }
}

void converImgU8toYCbCr( ImgU8 &src, ImgYCbCr &out )
{
    if ( src.depth < 3 )
        return;

    initImgYCbCr( out, src.width, src.height, src.depth );

    unsigned imgsz = src.width * src.height;

    #pragma omp parallel for
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fR = (float)src.buff[ ( cnt * src.depth ) + 0 ];
        float fG = (float)src.buff[ ( cnt * src.depth ) + 1 ];
        float fB = (float)src.buff[ ( cnt * src.depth ) + 2 ];

        // Y
        out.Y.buff[cnt] = ( 0.299f * fR ) +
                          ( 0.587f * fG ) +
                          ( 0.114f * fB );

        // Cb
        out.Cb.buff[cnt] = 128.f -
                           ( 0.1687f * fR ) -
                           ( 0.3313f * fG ) +
                           ( 0.5f * fB );

        // Cr
        out.Cr.buff[cnt] = 128.f +
                           ( 0.5f * fR ) -
                           ( 0.4187f * fG ) -
                           ( 0.0813f * fB );

        if ( ( src.depth == 4 ) && ( out.uA == true ) )
        {
            // Alpha
            out.A.buff[cnt] = (float)src.buff[ ( cnt * src.depth ) + 3 ];
        }
    }
}

void convertImgF32XtoImgU8( ImgF32* src, unsigned d, ImgU8 &out )
{
    if ( src == NULL )
        return;

    unsigned imgsz = src[0].width * src[0].height;

    out.width  = src[0].width;
    out.height = src[0].height;
    out.depth  = d;
    out.buff   = new unsigned char[ imgsz * d ];

    #pragma omp parallel for
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fY  = src[0].buff[cnt];
        float fCb = src[1].buff[cnt] - 128.f;
        float fCr = src[2].buff[cnt] - 128.f;

        float fR  = MIN(255.f, fY + 45.f * fCr / 32.f);
        float fG  = MIN(255.f, fY - ( 11.f * fCb + 23.f * fCr ) / 32.f);
        float fB  = MIN(255.f, fY + 113.f * fCb / 64.f );

        // Red -> Green -> Blue ...
        out.buff[( cnt * d ) + 0] = (unsigned char)MAX( 0.f, fR );
        out.buff[( cnt * d ) + 1] = (unsigned char)MAX( 0.f, fG );
        out.buff[( cnt * d ) + 2] = (unsigned char)MAX( 0.f, fB );

        if ( d == 4 )
        {
            float fA = MIN(255.f, src[3].buff[cnt]);
            out.buff[( cnt * d ) + 3] = (unsigned char)MAX( 0.f, fA );
        }
    }
}

void convertYCbCrtoImgU8( ImgYCbCr &src, unsigned d, ImgU8* &out )
{
    out = new ImgU8;

    if ( out == NULL )
        return;

    unsigned imgsz = src.Y.width * src.Y.height;

    out->width  = src.Y.width;
    out->height = src.Y.height;
    out->depth  = d;
    out->buff   = new unsigned char[ imgsz * d ];

    if ( out->buff == NULL )
        return;

    #pragma omp parallel for
    for( unsigned cnt=0; cnt<imgsz; cnt++ )
    {
        float fY  = src.Y.buff[cnt];
        float fCb = src.Cb.buff[cnt];
        float fCr = src.Cr.buff[cnt];

        // Red -> Green -> Blue ...
        out->buff[( cnt * d ) + 0] = \
                (unsigned char)(fY + ( 1.402f * fCr ));
        out->buff[( cnt * d ) + 1] = \
                (unsigned char)(fY - ( 0.34414f * fCb ) - ( 0.71414 * fCr ));
        out->buff[( cnt * d ) + 2] = \
                (unsigned char)(fY + ( 1.772 * fCb ));
        if ( d == 4 )
        {
            out->buff[( cnt * d ) +3] = src.A.buff[cnt];
        }
    }
}

////////////////////////////////////////////////////////////////////////////////

void convolution99( ImgF32 &src, ImgF32 &dst, const KernelMat99 kernel, float bias )
{
    /* Expand the src image */
    ImgF32 src2;
    initImgF32( src2, src.width + 8, src.height + 8 );

    if ( src2.buff == NULL )
    {
        resetImgF32( src2 );
        return;
    }

    for ( unsigned row = 0; row<src2.height; row++ )
    {
        for ( unsigned col = 0; col<src2.width; col++ )
        {
            int tmpRow = (int)row - 4;
            int tmpCol = (int)col - 4;

            if ( tmpRow < 0 )
            {
                tmpRow = 0;
            }
            else
            if ( tmpRow >= src.height )
            {
                tmpRow = src.height - 1;
            }

            if ( tmpCol < 0 )
            {
                tmpCol = 0;
            }
            else
            if ( tmpCol >= src.width )
            {
                tmpCol = src.width - 1;
            }

            src2.buff[ row * src2.width + col ] = \
                                src.buff[ tmpRow * src.width + tmpCol ];
        }
    }

    /* Complete the Convolution Step */
    for ( unsigned row=0; row<dst.height; row++ )
    {
        for ( unsigned col=0; col<dst.width; col++ )
        {
            /* Convolution */
            float temp = 0;

            for ( unsigned x=0; x<9; x++ )
            {
                for ( unsigned y=0; y<9; y++ )
                {
                    unsigned pos = ( row  + x ) * src2.width + ( col + y );

                    temp += kernel[x][y] * src2.buff[pos];
                }
            }

            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.buff[ row * dst.width + col ] = temp;
        }
    }

    delete[] src2.buff;
}

void convolution11( ImgConv1Layers &src, ImgF32 &dst, const ConvKernel1 kernel, float bias )
{
    for ( unsigned row=0; row<dst.height; row++ )
    {
        for ( unsigned col=0; col<dst.width; col++ )
        {
            /* Process with each pixel */
            float temp = 0;

            for ( unsigned fc=0; fc<CONV1_FILTERS; fc++ )
            {
                temp += src[fc].buff[ row * src[fc].width + col ]
                        * kernel[fc];
            }

            temp += bias;

            /* Threshold */
            temp = (temp >= 0) ? temp : 0;

            dst.buff[ row * dst.width + col ] = temp;
        }
    }
}

void convolution55( ImgConv2Layers &src, ImgF32 &dst, const ConvKernel32_55 kernel, float bias )
{
    /* Expand the src image */
    ImgConv2Layers src2;
    initImgConvLayers( &src2[0],
                       src[0].width + 4,
                       src[0].height + 4,
                       CONV2_FILTERS );

    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        for ( unsigned row=0; row<src2[cnt].height; row++ )
        {
            for ( unsigned col=0; col<src2[cnt].width; col++ )
            {
                int tmpRow = (int)row - 2;
                int tmpCol = (int)col - 2;

                if (tmpRow < 0)
                {
                    tmpRow = 0;
                }
                else
                if (tmpRow >= src[cnt].height)
                {
                    tmpRow = src[cnt].height - 1;
                }

                if (tmpCol < 0)
                {
                    tmpCol = 0;
                }
                else
                if (tmpCol >= src[cnt].width)
                {
                    tmpCol = src[cnt].width - 1;
                }

                src2[cnt].buff[ row * src2[cnt].width + col ] = \
                        src[cnt].buff[ tmpRow * src[cnt].width + tmpCol ];
            }
        }
    }

    /* Complete the Convolution Step */
    #pragma omp parallel for
    for ( unsigned row=0; row<dst.height; row++ )
    {
        for ( unsigned col=0; col<dst.width; col++ )
        {
            float temp = 0;

            for ( unsigned i=0; i<CONV2_FILTERS; i++ )
            {
                double temppixel = 0;

                for ( unsigned y=0; y<5; y++ )
                {
                    for ( unsigned x=0; x<5; x++ )
                    {
                        unsigned pos = ( (row + y) * src2[i].width ) \
                                       + ( col + x );
                        temppixel += kernel[i][x][y] * src2[i].buff[pos];
                    }
                }

                temp += temppixel;
            }

            temp += bias;

			temp = MAX( temp, 0.f );
			temp = MIN( temp, 255.f );

            dst.buff[ row * dst.width + col ] = temp;
        }
    }

    discardConvLayers( &src2[0], CONV2_FILTERS );
}

void Convolution99x11( ImgF32& src, ImgF32* dst, const ConvKernel64_99 kernel99, \
                                                 const ConvKernel1 bias99, \
                                                 const ConvKernel21 kernel11, \
                                                 const ConvKernel2 bias11 )
{
    float    result = 0.f;
    unsigned height   = src.height;
    unsigned width    = src.width;
    unsigned row      = 0;
    unsigned col      = 0;
    float    temp[CONV1_FILTERS] = {0.f};

#ifdef DEBUG
    if ( ( height == 0 ) || ( width == 0 ) )
    {
        fprintf( stderr, 
                 "Convolution99x11: Invalid image size, width = %u, height = %u\n",
                 width, height );
    }
#endif

    // allocation required by macOS clang compatibility.
    unsigned rowsz = height + 9;
    unsigned* rowf = new unsigned[rowsz];
    if ( rowf == NULL )
        return;

    unsigned colsz = width + 9;
    unsigned* colf = new unsigned[colsz];
    if ( colf == NULL )
    {
        delete[] rowf;
        return;
    }

    /* Expand the src image */
    #pragma omp parallel for
    for (row = 0; row < rowsz-1; row++)
    {
        rowf[row] = uTrim32(0, height - 1, row - 4);
    }

    #pragma omp parallel for
    for (col = 0; col < colsz-1; col++)
    {
        colf[col] = uTrim32(0, width - 1, col - 4);
    }

    /* Complete the Convolution Step */
    /* TODO : need to be optimized with OpenMP */
    for (row = 0; row < height; row++)
    {
        for (col = 0; col < width; col++)
        {
            for (unsigned k = 0; k < CONV1_FILTERS; k++)
            {
                /* Convolution */
                temp[k] = 0.f;

                for (unsigned i = 0; i < 9; i++)
                {
                    for (unsigned j = 0; j < 9; j++)
                    {
                        temp[k] += (float)(kernel99[k][i][j]) \
                                   * float(src.buff[ rowf[row + i] * width + colf[col + j] ]);
                    }
                }

                temp[k] += bias99[k];

                /* Threshold */
                temp[k] = (temp[k] < 0.f) ? 0.f : temp[k];
            }

            /* Process with each pixel */
            for (unsigned k = 0; k < CONV2_FILTERS; k++)
            {
                result = 0.0;

                for (unsigned i = 0; i < CONV1_FILTERS; i++)
                {
                    result += temp[i] * kernel11[k][i];
                }
                result += bias11[k];

                /* Threshold */
                result = (result < 0.f) ? 0.f : result;

                dst[k].buff[row * width + col] = result;
            }
        }
    }

    delete[] rowf;
    delete[] colf;
}

int doSRCNN( const unsigned char* refbuff,
             unsigned w, unsigned h, unsigned d,
             float muliply,
             unsigned char* &outbuff,
             unsigned &outbuffsz,
             unsigned char** convbuff,
             unsigned* convbuffsz )
{
    int retval = -100;

    // -------------------------------------------------------------
    // Convert RGB to Y-Cb-Cr
    //
    // warning: imgSrc is referenced, don't remove from memory !
    libsrcnn::ImgU8     imgSrc = { w ,h ,d, (unsigned char*)refbuff };
    libsrcnn::ImgYCbCr  imgYCbCr;

    converImgU8toYCbCr( imgSrc, imgYCbCr );

#ifdef DEBUG_COLORSAPCE
    saveImgYCbCr( &imgYCbCr, "debugimg" );
#endif

    /* --
     * Resize the Y Channel with Bicubic Interpolation,
     * Other layers just doing linear interpolation.
     */

    libsrcnn::ImgF32 imgResized[4];
    const float* refimgbuf[4] = { imgYCbCr.Y.buff,
                                  imgYCbCr.Cb.buff,
                                  imgYCbCr.Cr.buff,
                                  imgYCbCr.A.buff };

    unsigned rs_w = imgYCbCr.Y.width  * muliply;
    unsigned rs_h = imgYCbCr.Y.height * muliply;

    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<d; cnt++ )
    {
        imgResized[cnt].width  = rs_w;
        imgResized[cnt].height = rs_h;
        imgResized[cnt].depth  = 1;
        imgResized[cnt].buff   = NULL;

        FRAWGenericFilter* rszfilter;

        if ( cnt == 0 )
        {
            switch( libsrcnn::intp_filter )
            {
                default:
                case SRCNNF_Nearest:
                    rszfilter = new FRAWBoxFilter;
                    break;

                case SRCNNF_Bilinear:
                    rszfilter = new FRAWBilinearFilter;
                    break;

                case SRCNNF_Bicubic:
                    rszfilter = new FRAWBicubicFilter;
                    break;

                case SRCNNF_Lanczos3:
                    rszfilter = new FRAWLanczos3Filter;
                    break;

                case SRCNNF_Bspline:
                    rszfilter = new FRAWBSplineFilter;
                    break;
            }
        }
        else
        {
            switch( libsrcnn::intp_filter )
            {
                case SRCNNF_Nearest:
                    rszfilter = new FRAWBoxFilter;
                    break;

                default:
                case SRCNNF_Bilinear:
                    rszfilter = new FRAWBilinearFilter;
                    break;
            }
        }

        FRAWResizeEngine rsze( rszfilter );

        rsze.scale( refimgbuf[cnt],
                    imgYCbCr.Y.width,
                    imgYCbCr.Y.height,
                    rs_w,
                    rs_h,
                    &imgResized[cnt].buff );

        delete rszfilter;
    }

    // Release splitted image of Y-Cb-Cr --
    discardImgYCbCr( imgYCbCr );

#ifdef DEBUG
    printf("rY:");
    saveImgF32( &imgResized[0], "resized_Y.png" );
    printf("rCb:");
    saveImgF32( &imgResized[1], "resized_Cb.png" );
    printf("rCr:");
    saveImgF32( &imgResized[2], "resized_Cr.png" );
    if ( d == 4 )
    {
        printf("rA:");
        saveImgF32( &imgResized[3], "resized_A.png" );
    }
#endif

#ifdef NEW_FAST_I_II_LAYERS
    /******************* Fast I + II Layer *******************/

    libsrcnn::ImgConv2Layers imgConv2;

    libsrcnn::initImgConvLayers( imgConv2,
                                 imgResized[0].width,
                                 imgResized[0].height,
                                 CONV2_FILTERS );
#ifdef DEBUG
    printf( "initImgConvLayers, imgConv2 = %p, imgResized[0].width = %u, imgResized[0].height = %u, %u\n",
            imgConv2, imgResized[0].width, imgResized[0].height, CONV2_FILTERS );
    fflush( stdout );
#endif /// of DEBUG

    /* PERFORMANCE ISSUE !!
       Convolution99x11 saves memory than separated 99 and 11 convolution,
       But no way to apply OpenMP MPI for now.
    */
    Convolution99x11( imgResized[0],
                      imgConv2, weights_conv1_data, 
                      biases_conv1, weights_conv2_data, 
                      biases_conv2 );

    #ifdef DEBUG
        printf("new memory saving I & II layers ..\n" );
        fflush( stdout );

        #pragma omp parallel for
        for ( unsigned cnt=0; cnt<CONV2_FILTERS; cnt++ )
        {
            char strtmp[80] = {0};
            snprintf( strtmp, 80, "new_conv2_%u.png", cnt );
            printf( "Writing %s\n", strtmp ); fflush( stdout );
            saveImgF32( &imgConv2[cnt], strtmp );
        }
    #endif
#else
    /******************* The First Layer *******************/

    libsrcnn::ImgConv1Layers imgConv1;

    libsrcnn::initImgConvLayers( imgConv1,
                                 imgResized[0].width,
                                 imgResized[0].height,
                                 CONV1_FILTERS );
    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<CONV1_FILTERS; cnt++)
    {
        libsrcnn::convolution99( imgResized[0],
                                 imgConv1[cnt],
                                 weights_conv1_data[cnt],
                                 biases_conv1[cnt] );
    }

#ifdef DEBUG
    for ( unsigned cnt=0; cnt<CONV1_FILTERS; cnt++ )
    {
        char strtmp[80] = {0};
        snprintf( strtmp, 80, "conv1_%u.png", cnt );
        saveImgF32( &imgConv1[cnt], strtmp );
    }
#endif

    /******************* The Second Layer *******************/

    libsrcnn::ImgConv2Layers imgConv2;

    libsrcnn::initImgConvLayers( imgConv2,
                                 imgResized[0].width,
                                 imgResized[0].height,
                                 CONV2_FILTERS );
    #pragma omp parallel for
    for ( unsigned cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        libsrcnn::convolution11( imgConv1,
                                 imgConv2[cnt],
                                 weights_conv2_data[cnt],
                                 biases_conv2[cnt]);
    }

#ifdef DEBUG
    for ( unsigned cnt=0; cnt<CONV2_FILTERS; cnt++ )
    {
        char strtmp[80] = {0};
        snprintf( strtmp, 80, "conv2_%u.png", cnt );
        // saveImgF32( &imgConv2[cnt], strtmp );
    }
#endif /// of DEBUG
#endif /// of NEW_FAST_I_II_LAYERS

    /******************* The Third Layer *******************/

    libsrcnn::ImgF32 imgConv3;

    libsrcnn::initImgF32( imgConv3,
                          imgResized[0].width,
                          imgResized[0].height );

    libsrcnn::convolution55( imgConv2, imgConv3,
                             weights_conv3_data,
                             biases_conv3 );

#ifdef DEBUG
    saveImgF32( &imgConv3, "conv3.png" );
#endif

    // ---------------------------------------------------------
	// Copy Third layer result to Y channel.

	unsigned convsz = imgResized[0].width \
                      * imgResized[0].height * sizeof( float );
	memcpy( imgResized[0].buff, imgConv3.buff, convsz );

    /* Convert the image from YCrCb to RGB Space */
    libsrcnn::ImgU8 imgRGB;
    libsrcnn::convertImgF32XtoImgU8( imgResized, d, imgRGB );

    // discard used image of Resized Y-Cr-Cb.
    libsrcnn::discardConvLayers( imgResized, d );

    // discard used buffers ..
#ifndef NEW_FAST_I_II_LAYERS
    libsrcnn::discardConvLayers( &imgConv1[0], CONV1_FILTERS );
#endif /// of ! NEW_FAST_I_II_LAYERS
    libsrcnn::discardConvLayers( &imgConv2[0], CONV2_FILTERS );

    if ( imgRGB.buff != NULL )
    {
        outbuffsz = imgRGB.width * imgRGB.height * imgRGB.depth;
        outbuff = new unsigned char[ outbuffsz ];
        if ( outbuff != NULL )
        {
            memcpy( outbuff, imgRGB.buff, outbuffsz );
            retval = 0;
        }
        else
        {
            retval = -11;
        }

        resetImgU8( imgRGB );
    }

    if ( ( convbuff != NULL ) && ( convbuffsz != NULL ) )
    {
        if ( ( imgConv3.buff != NULL ) && ( retval == 0 ) )
        {
            unsigned bsz = imgConv3.width * imgConv3.height;
            unsigned char* buff = new unsigned char[ bsz ];
            if ( buff != NULL )
            {
				#pragma omp parallel for
				for( unsigned cnt=0; cnt<bsz; cnt++ )
                {
					buff[ cnt ] = (unsigned char)imgConv3.buff[ cnt ];
				}

				*convbuff   = buff;
				*convbuffsz = bsz;

                retval = 0;
            }
            else
            {
                retval = -12;
            }

            resetImgF32( imgConv3 );
        }
    }
    else
    if ( imgConv3.buff != NULL )
    {
        resetImgF32( imgConv3 );
    }

    return retval;
}
////////////////////////////////////////////////////////////////////////////////

}; /// of namespace libsrcnn

////////////////////////////////////////////////////////////////////////////////

void DLL_PUBLIC ConfigureFilterSRCNN( SRCNNFilterType ftype, bool stepscale )
{
    if ( libsrcnn::intp_filter != ftype )
    {
        libsrcnn::intp_filter = ftype;
    }

    if ( libsrcnn::intp_stepscale != stepscale )
    {
        libsrcnn::intp_stepscale = stepscale;
    }
}

int DLL_PUBLIC ProcessSRCNN( const unsigned char* refbuff,
                             unsigned w, unsigned h, unsigned d,
                             float multiply,
                             unsigned char* &outbuff,
                             unsigned &outbuffsz,
                             unsigned char** convbuff,
                             unsigned* convbuffsz )
{
    if ( ( refbuff == NULL ) || ( w == 0 ) || ( h == 0 ) || ( d == 0 ) )
        return -1;

#ifdef DEBUG
    printf( "ProcessSRCNN( w=%u, h=%d, d=%u, m=%f\n",
            w, h ,d, multiply );
    fflush( stdout );
#endif

    float m_w = (float)w * multiply;
    float m_h = (float)h * multiply;

    if ( ( m_w <= 0.f ) || ( m_h <= 0.f ) )
    {
        return -2;
    }

    int retval = -100;

    if ( libsrcnn::intp_stepscale == false )
    {
        retval = libsrcnn::doSRCNN( refbuff,
                                    w, h, d,
                                    multiply,
                                    outbuff,
                                    outbuffsz,
                                    convbuff,
                                    convbuffsz );
    }
    else
    {
        // Calc multiply by factor 2.0...
        float lf   = fmodf( multiply, 2.f );
        int repeat = (int)(multiply / 2.f);

        const unsigned char* rbuff = refbuff;
        unsigned char* obuff = NULL;
        unsigned obuffsz = 0;
        unsigned char** cbuff = NULL;
        unsigned* cbuffsz = NULL;
        unsigned sw = w;
        unsigned sh = h;

        if ( lf > 0.f )
        {
            repeat++;
        }

        for( int cnt=0; cnt<repeat; cnt++ )
        {
            float curmf = 2.0f;

            if ( cnt + 1 == repeat )
            {
                curmf = ( (float)w * multiply ) / (float)sw;

                if ( ( curmf == 0.f ) || ( curmf == 1.0f ) )
                    break;
            }

            obuffsz = sw * sh * d;

            if ( cnt > 0 )
            {
                if ( cnt > 1 )
                {
                    if ( rbuff != NULL )
                        delete[] rbuff;
                }

                rbuff = obuff;
                obuff = NULL;
            }

            if ( cnt + 1 == repeat )
            {
                cbuff   = convbuff;
                cbuffsz = convbuffsz;
            }

            retval = libsrcnn::doSRCNN( rbuff,
                                        sw, sh, d,
                                        curmf,
                                        obuff,
                                        obuffsz,
                                        cbuff,
                                        cbuffsz );

            if ( retval != 0 )
            {
                delete[] rbuff;
                rbuff = NULL;
                break;
            }

            if ( repeat > 1 )
            {
                sw *= curmf;
                sh *= curmf;
            }
        }

        if ( ( repeat > 1 ) && ( rbuff != NULL ) )
        {
            delete[] rbuff;
        }

        outbuff = obuff;
        outbuffsz = obuffsz;
        convbuff = cbuff;
    }

    return retval;
}
