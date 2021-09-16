#pragma once

#define DLLEXPORT __declspec(dllexport)

#ifdef __cplusplus
extern "C" {
#endif
  DLLEXPORT const char* imageToJsonString(const char* imgPath);

#ifdef __cplusplus
}
#endif