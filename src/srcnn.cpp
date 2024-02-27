#include <unistd.h>
#ifndef NO_OMP
#include <omp.h>
#endif

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cstdint>

#include <setjmp.h>

extern "C" 
{
#include <jpeglib.h>
#include <png.h>
}

#include <string>

#include "libsrcnn.h"
#include "tick.h"
#include "resource.h"
#include "minmax.h"

////////////////////////////////////////////////////////////////////////////////

using namespace std;

////////////////////////////////////////////////////////////////////////////////

struct jpg_err_mgr 
{
    struct jpeg_error_mgr pub;
    jmp_buf setjmp_buffer;
};

struct img_st
{
    unsigned w;
    unsigned h;
    unsigned d;
    uint8_t* data;
};

typedef struct _jsrc_mgr
{
    struct jpeg_source_mgr pub;
    const uint8_t* data;
    const uint8_t* s;
} jsrc_mgr;

typedef jsrc_mgr* jsrc_mgr_ptr;

////////////////////////////////////////////////////////////////////////////////
extern "C"
{
    static void jpg_err_hndlr( j_common_ptr dinfo )
    {
        jpg_err_mgr* myerr = (jpg_err_mgr*)dinfo->err;
        longjmp( myerr->setjmp_buffer, 1 );
    }

    static void jpg_output_hndlr( j_common_ptr jp )
    {
        // do nothing.
    }

    static void jpg_init_source( j_decompress_ptr cinfo )
    {
        jsrc_mgr_ptr src = (jsrc_mgr_ptr)cinfo->src;
        src->pub.next_input_byte = src->data;
        src->pub.bytes_in_buffer = src->s - src->data;
    }

    static boolean jpg_fill_input_buffer( j_decompress_ptr cinfo )
    {
        jsrc_mgr_ptr src = (jsrc_mgr_ptr)cinfo->src;
        size_t nbytes = 4096;
        src->pub.next_input_byte = src->data;
        src->s += nbytes;
        return TRUE;
    }

    static void jpg_term_source( j_decompress_ptr cinfo )
    {
        return;
    }

    static void jpg_skip_input_data( j_decompress_ptr cinfo, long num_bytes )
    {
        jsrc_mgr_ptr src = (jsrc_mgr_ptr)cinfo->src;
        if ( num_bytes > 0 )
        {
            while( num_bytes > (long)src->pub.bytes_in_buffer )
            {
                num_bytes -= (long)src->pub.bytes_in_buffer;
                jpg_fill_input_buffer( cinfo );
            }
            src->pub.next_input_byte += num_bytes;
            src->pub.bytes_in_buffer -= num_bytes;
        }
    }
} /// of extern "C" { ..

static void jpg_unprotected_mem_src( j_decompress_ptr cinfo, const uint8_t* data )
{
    jsrc_mgr_ptr src = new jsrc_mgr;
    if ( src != NULL )
    {
        cinfo->src = &(src->pub);

        src->data = data;
        src->s = data;
        src->pub.init_source = jpg_init_source;
        src->pub.fill_input_buffer = jpg_fill_input_buffer;
        src->pub.skip_input_data = jpg_skip_input_data;
        src->pub.resync_to_restart = jpeg_resync_to_restart;
        src->pub.term_source = jpg_term_source;
        src->pub.bytes_in_buffer = 0;
        src->pub.next_input_byte = NULL;
    }
}

void rem_img( img_st*& img )
{
    if ( img != NULL )
    {
        if ( img->data != NULL )
        {
            delete[] img->data;
        }
        delete img;
        img = NULL;
    }
}

int32_t load_png( const char* fn, img_st*& img )
{
    FILE* fp = fopen( fn, "rb" );

    if ( fp == NULL )
    {
        return -1;
    }

    int channels = 0;
    png_structp pp;
    png_infop info = 0;
    png_bytep* rows = NULL;
   
    pp = png_create_read_struct( PNG_LIBPNG_VER_STRING, NULL, NULL, NULL );
    if ( pp == NULL )
    {
        return -1;
    }   

    info = png_create_info_struct( pp );
    if ( info == NULL )
    {
        png_destroy_read_struct( &pp, NULL, NULL );
        return -1;
    }

    if ( setjmp( png_jmpbuf( pp ) ) )
    {
        png_destroy_read_struct( &pp, &info, NULL );
        fclose( fp );
        return -1;
    }

    png_init_io( pp, fp );
    png_read_info( pp, info );

    uint32_t png_coltype = png_get_color_type( pp, info );

    if ( png_coltype  == PNG_COLOR_TYPE_PALETTE )
    {
        png_set_palette_to_rgb( pp );
    }
    else
    if (  png_coltype == PNG_COLOR_TYPE_GRAY && png_get_bit_depth( pp, info ) < 8 )
    {
        png_set_expand_gray_1_2_4_to_8( pp );
    }
   
    if ( png_coltype & PNG_COLOR_MASK_COLOR )
    {
        channels = 3;
    }
    else
    {
        channels = 1;
    }

    int num_trans = 0;
    png_get_tRNS( pp, info, 0, &num_trans, 0 );
    if ( ( png_coltype & PNG_COLOR_MASK_ALPHA ) || ( num_trans != 0 ) )
    {
        channels++;
    }

    img = new img_st;
    if ( img == NULL )
    {
        png_destroy_read_struct( &pp, &info, NULL );
        fclose( fp );
        return -1;
    }

    img->w = png_get_image_width( pp, info );
    img->h = png_get_image_height( pp, info );
    img->d = channels;

    if ( png_get_bit_depth( pp, info ) < 8 )
    {
        png_set_packing( pp );
        png_set_expand( pp );
    }
    else
    if ( png_get_bit_depth( pp, info ) == 16 )
    {
        png_set_strip_16( pp );
    }

    if ( png_get_valid( pp, info, PNG_INFO_tRNS ) )
    {
        png_set_tRNS_to_alpha( pp );
    }

    img->data = new uint8_t[ img->w * img->h * img->d ];
    rows = new png_bytep[ img->h ];

    for( size_t cnt=0; cnt<img->h; cnt++ )
    {
        rows[ cnt ] = (png_bytep)( img->data + ( cnt * img->w * img->d ) );
    }

    for( size_t cnt = png_set_interlace_handling( pp ); cnt > 0; cnt-- )
    {
        png_read_rows( pp, rows, NULL, img->h );
    }

    if( channels == 4 ) 
    {
        for( size_t cnt=0; cnt<img->h; cnt++ )
        {
            for( size_t cnt2=0; cnt2<img->w; cnt2++ )
            {
                img->data[ ( cnt * img->w * img->d ) + ( cnt2 * img->d ) + 3 ] = 255;
            }
        }
    }

    delete[] rows;
    png_read_end( pp, info );
    png_destroy_read_struct( &pp, &info, NULL );
    fclose( fp );

    return 0;
}

int32_t load_jpg( const char* fn, img_st*& img )
{
    FILE* fp = fopen( fn, "rb" );

    if ( fp == NULL )
    {
        return -1;
    }

    jpeg_decompress_struct  dinfo;
    jpg_err_mgr             jerr;
    JSAMPROW                row;

    char* max_finish_decomp_err = NULL;
    char* max_destroy_decomp_err = NULL;

    dinfo.err                = jpeg_std_error( (jpeg_error_mgr*)&jerr );
    jerr.pub.error_exit     = jpg_err_hndlr;
    jerr.pub.output_message = jpg_output_hndlr;

    max_finish_decomp_err = (char*)malloc(1);
    max_destroy_decomp_err = (char*)malloc(1);
    *max_finish_decomp_err = 10;
    *max_destroy_decomp_err = 10;

    if ( setjmp( jerr.setjmp_buffer ) )
    {
        const char* name = "<unnamed>";
        if ( fn ) name = fn;

        if ( (*max_finish_decomp_err)-- > 0 )
            jpeg_finish_decompress( &dinfo );
        
        if ( ( *max_destroy_decomp_err )-- > 0 )
            jpeg_finish_decompress( &dinfo );

        fclose( fp );

        free( max_finish_decomp_err );
        free( max_destroy_decomp_err );

        return -1;
    }

    jpeg_create_decompress( &dinfo );

    img = new img_st;

    if ( img == NULL )
    {
        fclose( fp );

        free( max_finish_decomp_err );
        free( max_destroy_decomp_err );
       
        return -1;
    }

    jpeg_stdio_src( &dinfo, fp );
    jpeg_read_header( &dinfo, TRUE );

    dinfo.quantize_colors = (boolean)FALSE;
    dinfo.out_color_space = JCS_RGB;
    dinfo.out_color_components = 3;
    dinfo.output_components = 3;

    jpeg_calc_output_dimensions( &dinfo );
    img->w = dinfo.output_width;
    img->h = dinfo.output_height;
    img->d = dinfo.output_components;

    size_t imgsz = img->w * img->h * img->d;
    img->data = new uint8_t[ imgsz ];

    if ( img->data == NULL )
    {
        rem_img( img );
        fclose( fp );

        free( max_finish_decomp_err );
        free( max_destroy_decomp_err );
    }
    memset( img->data, 0, imgsz );

    jpeg_start_decompress( &dinfo );
    while( dinfo.output_scanline < dinfo.output_height )
    {
        row = (JSAMPROW)( img->data + ( dinfo.output_scanline * img->w * img->d ) );
        jpeg_read_scanlines( &dinfo, &row, 1 );
    }

    jpeg_finish_decompress( &dinfo );
    jpeg_destroy_decompress( &dinfo );
    
    free( max_destroy_decomp_err );
    free( max_finish_decomp_err );

    fclose( fp );

    return 0;
}

static img_st* newimg( const uint8_t* d, unsigned w, unsigned h, unsigned c )
{
    img_st* img = new img_st;
    if ( img != NULL )
    {
        img->w = w;
        img->h = h;
        img->d = c;
        img->data = new uint8_t[ w * h * c ];
        if ( img->data != NULL )
        {
            memcpy( img->data, d, w * h * c );
        }
        else
        {
            rem_img( img );
            img = NULL;
        }
    }

    return img;
}   

bool convImage( img_st* src, img_st*& dst )
{
    if ( src != NULL )
    {
        unsigned img_w = src->w;
        unsigned img_h = src->h;
        unsigned img_d = src->d;
        unsigned imgsz = img_w * img_h;
        uint8_t* cdata = NULL;

        switch( img_d )
        {
            case 1: /// single gray
            {
                const uint8_t* pdata = (const uint8_t*)src->data;
                cdata = new uint8_t[ imgsz * 3 ];
                if ( cdata != NULL )
                {
                    #pragma omp parallel for
                    for( unsigned cnt=0; cnt<imgsz; cnt++ )
                    {
                        cdata[ cnt*3 + 0 ] = pdata[ cnt ];
                        cdata[ cnt*3 + 1 ] = pdata[ cnt ];
                        cdata[ cnt*3 + 2 ] = pdata[ cnt ];
                    }

                    dst = newimg( cdata, img_w, img_h, 3 );

                    if ( dst != NULL )
                    {
                        return true;
                    }
                }
            }
            break;

            case 2: /// Must be RGB565
            {
                const uint16_t* pdata = (const uint16_t*)src->data;
                cdata = new uint8_t[ imgsz * 3 ];
                if ( cdata != NULL )
                {
                    #pragma omp parallel for
                    for( unsigned cnt=0; cnt<imgsz; cnt++ )
                    {
                        cdata[ cnt*3 + 0 ] = ( pdata[ cnt ] & 0xF800 ) >> 11;
                        cdata[ cnt*3 + 1 ] = ( pdata[ cnt ] & 0x07E0 ) >> 5;
                        cdata[ cnt*3 + 2 ] = ( pdata[ cnt ] & 0x001F );
                    }

                    dst = newimg( cdata, img_w, img_h, 3 );

                    if ( dst != NULL )
                    {
                        return true;
                    }
                }
            }
            break;
            
            default:
                {
                    dst = newimg( src->data, img_w, img_h, img_d );

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

int testImageFile( const char* imgfp )
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

                const uint8_t jpghdr[3] = { 0xFF, 0xD8, 0xFF };

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
            }

            fclose( fp );
        }
    }

    return reti;
}

bool savetopng( img_st* imgcached, const char* fpath )
{
    if ( imgcached == NULL )
        return false;

    // prevent from wrong or unsupoorted image.
    if ( ( imgcached->w == 0 ) || ( imgcached->h == 0 )  || ( imgcached->d == 2 ) )
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
                png_uint_32 mx = imgcached->w;
                png_uint_32 my = imgcached->h;
                png_uint_32 pd = imgcached->d;

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
                    const char* buf = (const char*)imgcached->data;
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
    printf( "%s : Neural Network Super Resolution Scaler, version %s\n", 
            file_me.c_str(),
            APP_VERSION_STR );
    printf( "(C)Copyrighted 2024 Raphael Kim | " );
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
    printf( "                      *  B-Spline is efficient and good for low resolution images.\n" );
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
 
    img_st* imgSrc = NULL;
    size_t imgsz = 0;
    
    int imgtype = testImageFile( file_src.c_str() );
    if ( imgtype > 0 )
    {
        printf( "\n" );
        printf( "- Image loaded type : ");
        
        switch( imgtype )
        {
            case 1: /// JPEG
                printf( "JPEG | ");
                load_jpg( file_src.c_str(), imgSrc );
                break;

            case 2: /// PNG
                printf( "PNG | " );
                load_png( file_src.c_str(), imgSrc );
                break;
               
            default: /// unknown...
                printf( "Unsupported !\n" );
                break;
        }
        
        if ( imgSrc != NULL )
        {
            printf( "%u x %u x %u bytes\n", imgSrc->w, imgSrc->h, imgSrc->d );
            fflush( stdout );
        }
    }
    
    if ( imgSrc != NULL )
    {
        const uint8_t* refbuff = imgSrc->data;
        unsigned     ref_w   = imgSrc->w;
        unsigned     ref_h   = imgSrc->h;
        unsigned     ref_d   = imgSrc->d;
        uint8_t*     outbuff = NULL;
        unsigned     outsz   = 0;
        uint8_t*     convbuff = NULL;
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
            
            printf( "Test Ok, took %.3f seconds.\n", 
                    (float)(tick1 - tick0 ) / 1000.f );
        
            img_st* imgDump = newimg( outbuff, new_w, new_h, ref_d );
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
                
                rem_img( imgDump );
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
         
            // save convolution result.
            img_st* imgConv = newimg( convbuff, new_w, new_h, ref_d );
            if ( imgConv != NULL )
            {
                printf( "- Saving convolution result to %s ... ", file_cov.c_str() );
                
                if ( savetopng( imgConv, file_cov.c_str() ) == true )
                {
                    printf( "Ok.\n" );
                }
                else
                {
                    printf( "Failure.\n" );
                }
                
                fflush( stdout );
                
                rem_img( imgConv );
            }
        }                  
    }
    else
    {
        printf( "- Failed to load image.\n" );
    }
   
    return 0;
}
