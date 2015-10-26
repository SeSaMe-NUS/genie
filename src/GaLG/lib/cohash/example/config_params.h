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

#include <vector>
#include <string>

#include <img_tga.h>

struct ConfigParams
{
  std::vector<std::string> sparams;
  std::string image_name;
  unsigned int num_keys;
  unsigned int num_extra;
  float dens;
  float rate_non_valid_keys;
  unsigned int seed;
  bool sorted_access;
  bool rand_num_mode;
  bool image_mode;
  bool access_null_keys;
  bool is_set;

  bool rh_rand_hash_state;
  bool rh_coh_hash_state;
  bool coh_hash;
  bool rand_hash;
  //--------------------------------------------
  float rh_coh_hash_build_time;
  float rh_coh_hash_access_time;
  float rh_rand_hash_build_time;
  float rh_rand_hash_access_time;
  //--------------------------------------------
  float rh_coh_hash_build_keys_per_sec;
  float rh_coh_hash_access_keys_per_sec;
  float rh_rand_hash_build_keys_per_sec;
  float rh_rand_hash_access_keys_per_sec;
  //--------------------------------------------
  unsigned int u3D_w;
  unsigned int u3D_h;
  unsigned int u3D_d;
  unsigned int u2D_w;
  unsigned int u2D_h;
  Texture *tex;
};