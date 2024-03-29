#ifdef FORTESTINGBIN

#if defined(_WIN32) || defined(WIN32)
    #include <windows.h>
#endif

#include <unistd.h>
#ifndef NO_OMP
#include <omp.h>
#endif
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

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

#include <string>

#include "libsrcnn.h"
#include "fl_imgtk.h"
#include "tick.h"
#include "resource.h"
#include "minmax.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;

////////////////////////////////////////////////////////////////////////////////

bool convImage( Fl_RGB_Image* src, Fl_RGB_Image* &dst )
{
    if ( src != NULL )
    {
        unsigned img_w = src->w();
        unsigned img_h = src->h();
        unsigned img_d = src->d();
        unsigned imgsz = img_w * img_h;

        uchar* cdata = NULL;

        switch( img_d )
        {
            case 1: /// single gray
                {
                    const uchar* pdata = (const uchar*)src->data()[0];
                    cdata = new uchar[ imgsz * 3 ];
                    if ( cdata != NULL )
                    {
                        #pragma omp parallel for
                        for( unsigned cnt=0; cnt<imgsz; cnt++ )
                        {
                            cdata[ cnt*3 + 0 ] = pdata[ cnt ];
                            cdata[ cnt*3 + 1 ] = pdata[ cnt ];
                            cdata[ cnt*3 + 2 ] = pdata[ cnt ];
                        }

                        dst = new Fl_RGB_Image( cdata, img_w, img_h, 3 );

                        if ( dst != NULL )
                        {
                            return true;
                        }
                    }
                }
                break;

            case 2: /// Must be RGB565
                {
                    const unsigned short* pdata = (const unsigned short*)src->data()[0];
                    cdata = new uchar[ imgsz * 3 ];
                    if ( cdata != NULL )
                    {
                        #pragma omp parallel for
                        for( unsigned cnt=0; cnt<imgsz; cnt++ )
                        {
                            cdata[ cnt*3 + 0 ] = ( pdata[ cnt ] & 0xF800 ) >> 11;
                            cdata[ cnt*3 + 1 ] = ( pdata[ cnt ] & 0x07E0 ) >> 5;
                            cdata[ cnt*3 + 2 ] = ( pdata[ cnt ] & 0x001F );
                        }

                        dst = new Fl_RGB_Image( cdata, img_w, img_h, 3 );

                        if ( dst != NULL )
                        {
                            return true;
                        }
                    }
                }
                break;
                
            default:
                {
                    dst = (Fl_RGB_Image*)src->copy();

                    if ( dst != NULL )
                    {
                        return true;
                    }
                }
                break;
        }
    }

    return false;
}

int testImageFile( const char* imgfp, uchar** buff,size_t* buffsz )
{
    int reti = -1;

    if ( imgfp != NULL )
    {
        FILE* fp = fopen( imgfp, "rb" );
        if ( fp != NULL )
        {
            fseek( fp, 0L, SEEK_END );
            size_t flen = ftell( fp );
            fseek( fp, 0L, SEEK_SET );

            if ( flen > 32 )
            {
                // Test
                char testbuff[32] = {0,};

                size_t rs = fread( testbuff, 1, 32, fp );
                fseek( fp, 0, SEEK_SET );

                const uchar jpghdr[3] = { 0xFF, 0xD8, 0xFF };

                // is JPEG ???
                if( strncmp( &testbuff[0], (const char*)jpghdr, 3 ) == 0 )
                {
                    reti = 1; /// JPEG.
                }
                else
                if( strncmp( &testbuff[1], "PNG", 3 ) == 0 )
                {
                    reti = 2; /// PNG.
                }
                else
                if( strncmp( &testbuff[0], "BM", 2 ) == 0 )
                {
                    reti = 3; /// BMP.
                }

                if ( reti > 0 )
                {
                    *buff = new uchar[ flen ];
                    if ( *buff != NULL )
                    {
                        rs = fread( *buff, 1, flen, fp );

                        if( buffsz != NULL )
                        {
                            *buffsz = flen;
                        }
                    }
                }
            }

            fclose( fp );
        }
    }

    return reti;
}

bool savetopng( Fl_RGB_Image* imgcached, const char* fpath )
{
    if ( imgcached == NULL )
        return false;

    // prevent from wrong or unsupoorted image.
    if ( ( imgcached->w() == 0 ) || ( imgcached->h() == 0 )  || ( imgcached->d() == 2 ) )
        return false;

    FILE* fp = fopen( fpath, "wb" );
    if ( fp == NULL )
        return false;

    png_structp png_ptr     = NULL;
    png_infop   info_ptr    = NULL;
    png_bytep   row         = NULL;

    png_ptr = png_create_write_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
    if ( png_ptr != NULL )
    {
        info_ptr = png_create_info_struct( png_ptr );
        if ( info_ptr != NULL )
        {
            if ( setjmp( png_jmpbuf( (png_ptr) ) ) == 0 )
            {
                png_uint_32 mx = imgcached->w();
                png_uint_32 my = imgcached->h();
                png_uint_32 pd = imgcached->d();
                // defualt PNG type is RGB 
                int ct = PNG_COLOR_TYPE_RGB;
                if ( pd == 1 )
                {
                    ct= PNG_COLOR_TYPE_GRAY;
                }
                else
                if ( pd == 4 )
                {
                    ct = PNG_COLOR_TYPE_RGBA;
                }

                png_init_io( png_ptr, fp );
                png_set_IHDR( png_ptr,
                              info_ptr,
                              mx,
                              my,
                              8,
                              ct,
                              PNG_INTERLACE_NONE,
                              PNG_COMPRESSION_TYPE_BASE,
                              PNG_FILTER_TYPE_BASE);

                png_write_info( png_ptr, info_ptr );

                row = new png_byte[ mx * pd ];

                if ( row != NULL )
                {
                    const char* buf = imgcached->data()[0];
                    png_uint_32 bque = 0;

                    for( png_uint_32 y=0; y<my; y++ )
                    {
                        for( png_uint_32 x=0; x<mx; x++ )
                        {
                            for( png_uint_32 dd=0; dd<pd; dd++ )
                            {
                                row[ (x*pd) + dd ] = buf[ bque + dd ];
                            }

                            bque += pd;
                        }

                        png_write_row( png_ptr, row );
                    }

                    png_write_end( png_ptr, NULL );

                    fclose( fp );

                    delete[] row;
                }

                png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
                png_destroy_write_struct(&png_ptr, (png_infopp)NULL);

                return true;
            }
        }
    }

    return false;
}

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////

static float    image_multiply  = 2.0f;
static bool     stepscale = false;
static bool     waitforakey = false;
static string   path_me;
static string   file_me;
static string   file_src;
static string   file_dst;
static string   file_cov;
static SRCNNFilterType filter_type = SRCNNF_Bicubic;

bool parseArgs( int argc, char** argv )
{
    for( int cnt=0; cnt<argc; cnt++ )
    {
        string strtmp = argv[ cnt ];
        size_t fpos   = string::npos;

        if ( cnt == 0 )
        {
            fpos = strtmp.find_last_of( "\\" );

            if ( fpos == string::npos )
            {
                fpos = strtmp.find_last_of( "/" );
            }

            if ( fpos != string::npos )
            {
                path_me = strtmp.substr( 0, fpos );
                file_me = strtmp.substr( fpos + 1 );
            }
            else
            {
                file_me = strtmp;
            }
        }
        else
        {
            if ( strtmp.find( "--scale=" ) == 0 )
            { 
                string strval = strtmp.substr( 8 );
                if ( strval.size() > 0 )
                {
                    float tmpfv = atof( strval.c_str() );
                    if ( tmpfv > 0.f )
                    {
                        image_multiply = tmpfv;
                    }
                }
            }
            else
            if ( strtmp.find( "--step" ) == 0 )
            {
                stepscale = true;
            }
            else
            if ( strtmp.find( "--filter=" ) == 0 )
            { 
                string strval = strtmp.substr( 9 );
                if ( strval.size() > 0 )
                {
                    int tmpi = atoi( strval.c_str() );
                    
                    switch( tmpi )
                    {
                        case 0:
                            filter_type = SRCNNF_Nearest;
                            break;
                            
                        case 1:
                            filter_type = SRCNNF_Bilinear;
                            break;
                            
                        default:
                        case 2:
                            filter_type = SRCNNF_Bicubic;
                            break;
                            
                        case 3:
                            filter_type = SRCNNF_Lanczos3;
                            break;
                            
                        case 4:
                            filter_type = SRCNNF_Bspline;
                            break;

                    }
                }
            }
            else
            if ( strtmp.find( "--waitakey" ) == 0 )
            {
                waitforakey = true;
            }
            else
            if ( file_src.size() == 0 )
            {
                file_src = strtmp;
            }
            else
            if ( file_dst.size() == 0 )
            {
                file_dst = strtmp;
            }
        }
    }
    
    if ( ( file_src.size() > 0 ) && ( file_dst.size() == 0 ) )
    {
        string convname = file_src;
        string srcext;
        
        // changes name without file extention.
        size_t posdot = file_src.find_last_of( "." );
        if ( posdot != string::npos )
        {
            convname = file_src.substr( 0, posdot );
            srcext   = file_src.substr( posdot );
        }
        
        convname += "_resized";
        if ( srcext.size() > 0 )
        {
            if ( ( srcext != ".png" ) || ( srcext != ".PNG" ) )
            {
                convname += ".png";
            }
            else
            {
                convname += srcext;
            }
        }
        
        file_dst = convname;
    }
    
    if ( ( file_src.size() > 0 ) && ( file_dst.size() > 0 ) )
    {
        string convname = file_src;
        string srcext;
        
        // changes name without file extention.
        size_t posdot = file_src.find_last_of( "." );
        if ( posdot != string::npos )
        {
            convname = file_src.substr( 0, posdot );
            srcext   = file_src.substr( posdot );
        }
        
        convname += "_convolution";
        if ( srcext.size() > 0 )
        {
            if ( ( srcext != ".png" ) || ( srcext != ".PNG" ) )
            {
                convname += ".png";
            }
            else
            {
                convname += srcext;
            }
        }
        
        file_cov = convname;

        return true;
    }
    
    return false;
}

const char* getPlatform()
{
    static char retstr[32] = {0};
#if defined(__WIN64)
    snprintf( retstr, 32, "Windows64" );
#elif defined(__WIN32)
    snprintf( retstr, 32, "Windows32" );
#elif defined(__APPLE__)
    snprintf( retstr, 32, "macOS" );
#elif defined(__linux__)
    snprintf( retstr, 32, "Linux" );
#else
    snprintF( retstr, 32, "unknown platform" );
#endif
    return retstr;
}

const char* getCompilerVersion()
{
    static char retstr[64] = {0};
#if defined(__MINGW64__)
    snprintf( retstr, 64,
              "MinGW-W64-%d.%d(%d.%d.%d)",
              __MINGW64_VERSION_MAJOR,
              __MINGW64_VERSION_MINOR,
              __GNUC__,
              __GNUC_MINOR__,
              __GNUC_PATCHLEVEL__ );
#elif defined(__MINGW32__)
    snprintf( retstr, 64,
              "MinGW-W32-%d.%d(%d.%d.%d)",
              __MINGW32_MAJOR_VERSION,
              __MINGW32_MINOR_VERSION, 
              __GNUC__,
              __GNUC_MINOR__,
              __GNUC_PATCHLEVEL__ );
#elif defined(__GNUC__)
    snprintf( retstr, 64,
              "GNU-GCC-%d.%d.%d",
              __GNUC__,
              __GNUC_MINOR__,
              __GNUC_PATCHLEVEL__ );
#else
    strncat( retstr, "unknown comiler", 64 );
#endif

    return retstr;
}

void printAbout()
{
    printf( "%s : libsrcnn testing program with FLTK-%d.%d.%d, ver %s\n", 
            file_me.c_str(),
            FL_MAJOR_VERSION,
            FL_MINOR_VERSION,
            FL_PATCH_VERSION,
            APP_VERSION_STR );
    printf( "(C)Copyrighted ~2023 Raphael Kim | " );
    printf( "build for %s, %s\n", getPlatform(), getCompilerVersion() );
    printf( "\n");
    fflush( stdout );   
}

void printUsage()
{
    printf( "  usage:\n" );
    printf( "      %s [source image file] [options] (output image file)\n", 
            file_me.c_str() );
    printf( "\n" );
    printf( "  options:\n" );
    printf( "      --scale=(ratio : 0.0<999...) : adjust size of output image.\n" );
    printf( "      --step                       : scaling by fator 2 steps.\n" );
    printf( "                                     * step scaling takes a lot of times.\n" );
    printf( "      --waitakey                   : wait for ENTER for end of job.\n" );
    printf( "      --filter=(0...4)             : Changes interpolation filter as ...\n" );
    printf( "                   0 = Nearest filter\n" );
    printf( "                   1 = Bilinear filter\n" );
    printf( "                   2 = Bicubic filter (default)\n" );
    printf( "                   3 = Lanzcos-3 filter\n" );
    printf( "                   4 = B-Spline filter\n" );
    printf( "\n" ); 
}

int main( int argc, char** argv )
{   
    if ( parseArgs( argc, argv ) == false )
    {
        printAbout();
        printUsage();
        return -1;
    }
    
    if ( image_multiply <= 0.0f )
    {
        printf( "- Error: scaling value under zero.\n" );
        return -2;
    }
    
    printf( "- Loading image : %s", file_src.c_str() );
    
    Fl_RGB_Image* imgTest = NULL;
    
    uchar* imgbuff = NULL;
    size_t imgsz = 0;
    
    int imgtype = testImageFile( file_src.c_str(), &imgbuff, &imgsz );
    if ( imgtype > 0 )
    {
        printf( "\n" );
        printf( "- Image loaded type : ");
        
        switch( imgtype )
        {
            case 1: /// JPEG
                printf( "JPEG | ");
                
                imgTest = new Fl_JPEG_Image( "JPGIMG",
                                            (const uchar*)imgbuff );
                break;

            case 2: /// PNG
                printf( "PNG | " );
                
                imgTest = new Fl_PNG_Image( "PNGIMAGE",
                                           (const uchar*)imgbuff, imgsz );
                break;

            case 3: /// BMP
                printf( "BMP | " );
                imgTest = fl_imgtk::createBMPmemory( (const char*)imgbuff, imgsz );
                break;
                
            default: /// unknown...
                printf( "Unsupported !\n" );
                break;
        }
        
        if ( imgTest != NULL )
        {
            printf( "%u x %u x %u bytes\n", imgTest->w(), imgTest->h(), imgTest->d() );
            fflush( stdout );
        }
        
        delete[] imgbuff;
        imgbuff = NULL;
        imgsz = 0;
    }
    
    if ( imgTest != NULL )
    {
        Fl_RGB_Image* imgRGB = NULL;
        
        convImage( imgTest, imgRGB );
        
        delete imgTest;
        
        if ( ( imgRGB->w() > 0 ) && ( imgRGB->h() > 0 ) 
             && ( imgRGB->d() >= 3 ) )
        {
            const uchar* refbuff = (const uchar*)imgRGB->data()[0];
            unsigned     ref_w   = imgRGB->w();
            unsigned     ref_h   = imgRGB->h();
            unsigned     ref_d   = imgRGB->d();
            uchar*       outbuff = NULL;
            unsigned     outsz   = 0;
            uchar*       convbuff = NULL;
            unsigned     convsz   = 0;
            
            printf( "- Scaling ratio : %.2f\n", image_multiply );
            if ( stepscale == true )
            {
                printf( "- Step scaling enabled, warning: takes a lot of times.\n" );
            }
            printf( "- Filter : ");
            switch( filter_type )
            {
                case SRCNNF_Nearest:
                    printf( "Nearest\n" );
                    break;
                    
                case SRCNNF_Bilinear:
                    printf( "Bilinear\n" );
                    break;
                    
                case SRCNNF_Bicubic:
                    printf( "Bicubic\n" );
                    break;
                    
                case SRCNNF_Lanczos3:
                    printf( "Lanczos3\n" );
                    break;
                    
                case SRCNNF_Bspline:
                    printf( "B-Spline\n" );
                    break;
            }
            
            ConfigureFilterSRCNN( filter_type, stepscale );
            fflush( stdout );
            
            printf( "- Processing SRCNN ... " );
            fflush( stdout );
            
            unsigned tick0 = tick::getTickCount();
            
            int reti = ProcessSRCNN( refbuff,
                                     ref_w,
                                     ref_h,
                                     ref_d,
                                     image_multiply,
                                     outbuff,
                                     outsz,
                                     &convbuff,
                                     &convsz );
            
            unsigned tick1 = tick::getTickCount();
			            
            if ( ( reti == 0 ) && ( outsz > 0 ) )
            {
                unsigned new_w = ref_w * image_multiply;
                unsigned new_h = ref_h * image_multiply;
                
                printf( "Test Ok, took %u ms.\n", tick1 - tick0 );
            
                Fl_RGB_Image* imgDump = new Fl_RGB_Image( outbuff, new_w, new_h, ref_d );
                if ( imgDump != NULL )
                {
                    printf( "- Saving resized result to %s ... ", file_dst.c_str() );
                    
                    if ( savetopng( imgDump, file_dst.c_str() ) == true )
                    {
                        printf( "Ok.\n" );
                    }
                    else
                    {
                        printf( "Failure.\n" );
                    }
                    
                    fflush( stdout );
                        
                    fl_imgtk::discard_user_rgb_image( imgDump );
                }
            }
            else
            {
                printf( "Failed, error code = %d\n", reti );
            }
            
            if ( ( reti == 0 ) && ( convsz > 0 ) )
            {
                unsigned new_w = ref_w * image_multiply;
                unsigned new_h = ref_h * image_multiply;
                            
                Fl_RGB_Image* imgDump = new Fl_RGB_Image( convbuff, new_w, new_h, 1 );
                if ( imgDump != NULL )
                {
                    printf( "- Saving convolution result to %s ... ", file_cov.c_str() );
                    
                    if ( savetopng( imgDump, file_cov.c_str() ) == true )
                    {
                        printf( "Ok.\n" );
                    }
                    else
                    {
                        printf( "Failure.\n" );
                    }
                    
                    fflush( stdout );
                        
                    fl_imgtk::discard_user_rgb_image( imgDump );
                }
            }           
            			
            delete imgRGB;
        }
        else
        {
            printf( "- Error: Unsupported image.\n" );
        }
    }
    else
    {
        printf( "- Failed to load image.\n" );
    }

    if ( waitforakey == true )
    {
        // let check memory leak before program terminated.
        printf( "- Input any number and press ENTER to terminate, check memory state.\n" );
        fflush( stdout );
        unsigned meaningless = 0;
        size_t rs = scanf( "%u", &meaningless );
    }
    
    return 0;
}

#endif /// of FORTESTINGBIN
