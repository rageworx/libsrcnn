#ifdef DEBUG
#ifndef __DEBUGTOOLH__
#define __DEBUGTOOLH__

void saveImgU8( void* img = NULL, const char* fname = NULL );
void saveImgF32( void* img = NULL, const char* fname = NULL );
void saveImgYCbCr( void* img = NULL, const char* fnameprefix = NULL );

#endif /// of __DEBUGTOOLH__
#endif /// of DEBUG