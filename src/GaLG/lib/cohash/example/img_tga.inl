/*
 *  (C) copyright  2011, Ismael Garcia, (U.Girona/ViRVIG, Spain & INRIA/ALICE, France)
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#include <img_tga.h>

//------------------------------------------------------------------------

const size_t size_uchar = sizeof(uchar);
const size_t size_sint = sizeof(sint);

//------------------------------------------------------------------------

Texture* loadTGA(char* fn)
{
  Texture* tga = NULL;
  FILE* fh = NULL;
  int md, t;

  /* Allocate memory for the info structure. */
  tga = (Texture *)malloc(sizeof(Texture));

  /* Open the file in binary mode. */
  fh = fopen(fn, "rb");

  /* Problem opening file? */
  if (fh == NULL)
  {
    fprintf(stderr, "Error: problem opening TGA file (%s).\n", fn);
  }
  else
  {
    tga = (Texture *)malloc(sizeof(Texture));

    // Load information about the tga, aka the header.
    {
      // Seek to the width.
      fseek(fh, 12, SEEK_SET);
      fread(&tga->w, size_sint, 1, fh);
     
      // Seek to the height.
      fseek(fh, 14, SEEK_SET);
      fread(&tga->h, size_sint, 1, fh);
     
      // Seek to the depth.
      fseek(fh, 16, SEEK_SET);
      fread(&tga->depth, size_sint, 1, fh);
    }
   
    // Load the actual image data.
    {
      // Mode = components per pixel.
      md = tga->depth / 8;

      // Total bytes = h * w * md.
      t = tga->h * tga->w * md;

      // Allocate memory for the image data.
      tga->data = (unsigned char *)malloc(size_uchar * t);

      // Seek to the image data.
      fseek(fh, 18, SEEK_SET);
      fread(tga->data, size_uchar, t, fh);

      // We're done reading.
      fclose(fh);

      // Mode 3 = RGB, Mode 4 = RGBA
      // TGA stores RGB(A) as BGR(A) so
      // we need to swap red and blue.
      if (md >= 3)
      {
        uchar aux;

        for (int i = 0; i < t; i+= md)
        {
                aux = tga->data[i];
                tga->data[i] = tga->data[i + 2];
                tga->data[i + 2] = aux;
        }
      }
    }
  }

  return tga;
}

//------------------------------------------------------------------------

void saveTGA(Texture *tga, char *filename) 
{
  FILE *Handle;
  unsigned char Header[18];

  Header[ 0] = 0;
  Header[ 1] = 0;
  Header[ 2] = 2;     // Uncompressed
  Header[ 3] = 0;
  Header[ 4] = 0;
  Header[ 5] = 0;
  Header[ 6] = 0;
  Header[ 7] = 0;
  Header[ 8] = 0;
  Header[ 9] = 0;
  Header[10] = 0;
  Header[11] = 0;
  Header[12] = (unsigned char) tga->w;  // dimensions
  Header[13] = (unsigned char) ((unsigned long) tga->w >> 8);
  Header[14] = (unsigned char) tga->h;
  Header[15] = (unsigned char) ((unsigned long) tga->h >> 8);
  Header[16] = 32;    // bits per pixel
  Header[17] = 8;

  Handle = fopen(filename, "wb");
  if(Handle == NULL) 
  {
    fprintf(stderr, "Error: can't open %s\n", filename);
  }
  fseek(Handle, 0, 0);
  fwrite(Header, 1, 18, Handle);
  fseek(Handle, 18, 0);

  for(unsigned int cswap = 0; cswap < tga->w * tga->h * (tga->depth / 8); cswap +=tga->depth / 8)
  {
    tga->data[cswap] ^= tga->data[cswap+2] ^= tga->data[cswap] ^= tga->data[cswap+2];
  }

  fwrite(tga->data, 4, tga->w * tga->h, Handle);

  for(unsigned int cswap = 0; cswap < tga->w * tga->h * (tga->depth / 8); cswap += tga->depth / 8)
  {
    tga->data[cswap] ^= tga->data[cswap+2] ^= tga->data[cswap] ^= tga->data[cswap+2];
  }

  fclose(Handle);
}

//------------------------------------------------------------------------

