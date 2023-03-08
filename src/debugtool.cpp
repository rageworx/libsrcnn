#ifdef DEBUG
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <algorithm>

#include <omp.h>

#include "debugtool.h"

#include <FL/Fl.H>
#include <FL/Fl_Window.H>
#include <FL/Fl_Box.H>
#include <FL/Fl_Image.H>
#include <FL/Fl_RGB_Image.H>
#include <FL/Fl_BMP_Image.H>
#include <FL/Fl_PNG_Image.H>
#include <FL/Fl_JPEG_Image.H>

#if defined(__linux__)
#include <png.h>
#else
#include <FL/images/png.h>
#endif

#include <fl_imgtk.h>

////////////////////////////////////////////////////////////////////////////////

using namespace std;

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
}ImgYCbCr;

////////////////////////////////////////////////////////////////////////////////

extern bool convImage( Fl_RGB_Image* src, Fl_RGB_Image* &dst );
extern bool savetocolorpng( Fl_RGB_Image* imgcached, const char* fpath );

////////////////////////////////////////////////////////////////////////////////

bool convertF32toU8( ImgF32* src, ImgU8 &dst )
{
    if ( src == NULL )
        return false;

    // don't care depth ...
    unsigned srcsz = src->width * src->height;

    dst.width  = src->width;
    dst.height = src->height;
    dst.buff   = new unsigned char[ srcsz ];

    if ( dst.buff == NULL )
        return false;

    // get max and min ...
    float fMin = 1.4013e-45f;
    float fMax = 0.f;

    for( unsigned cnt=0; cnt<srcsz; cnt++ )
    {
        if ( src->buff[ cnt ] > fMax )
        {
            fMax = src->buff[ cnt ];
        }
        else
        if ( src->buff[ cnt ] < fMin )
        {
            fMin = src->buff[ cnt ];
        }
    }

#ifdef DEBUG_COLORSPACE
    if ( fMin < 0.f )
    {
        printf( "Warning @ convertF32toU8(), Min float under zero : %.2f\n",
                fMin );
        fflush( stdout );
    }

    printf( "fMin:fMax=%.2f:%.2f", fMin, fMax );
#endif

    #pragma omp parallel for
    for( unsigned cnt=0; cnt<srcsz; cnt++ )
    {
        dst.buff[ cnt ] = ( unsigned char ) \
                          ( ( src->buff[ cnt ] / fMax ) * 255.f );
    }

    return true;
}

void saveImgU8( void* img, const char* fname )
{
    if ( ( img == NULL ) || ( fname == NULL ) )
        return;

    ImgU8* refimg = (ImgU8*)img;

    Fl_RGB_Image* imgTmp = new Fl_RGB_Image( refimg->buff,
                                             refimg->width,
                                             refimg->height,
                                             refimg->depth );
    if ( imgTmp != NULL )
    {
        Fl_RGB_Image* imgConv = NULL;
        if ( convImage( imgTmp, imgConv ) == true )
        {
            savetocolorpng( imgConv, fname );

            fl_imgtk::discard_user_rgb_image( imgConv );
        }

        // don't use discard_user ... () memory reference error.
        delete imgTmp;
    }
}

void saveImgF32( void* img, const char* fname )
{
    if ( ( img == NULL ) || ( fname == NULL ) )
        return;

    ImgF32* refimg = (ImgF32*)img;
    ImgU8 imgTmp = {0,0,1,NULL};

    if ( convertF32toU8( refimg, imgTmp ) == true )
    {
        saveImgU8( &imgTmp, fname );

        delete[] imgTmp.buff;
    }
}

void saveImgYCbCr( void* img, const char* fnameprefix )
{
    if ( ( img == NULL ) || ( fnameprefix == NULL ) )
        return;

    ImgYCbCr* refimg = (ImgYCbCr*)img;

    char strFnMap[1024] = {0};

    // Write Y
    printf("saveImgYCbCr(%s), Y:", fnameprefix ); fflush( stdout );
    sprintf( strFnMap, "%s_Y.png", fnameprefix );
    saveImgF32( &refimg->Y, strFnMap );

    // Write Cb
    printf("Cb:" ); fflush( stdout );
    sprintf( strFnMap, "%s_Cb.png", fnameprefix );
    saveImgF32( &refimg->Cb, strFnMap );

    // Write Cr
    printf("Cr:" ); fflush( stdout );
    sprintf( strFnMap, "%s_Cr.png", fnameprefix );
    saveImgF32( &refimg->Cr, strFnMap );

}

#endif /// of DEBUG
