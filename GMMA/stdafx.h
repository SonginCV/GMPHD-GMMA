// stdafx.h : 자주 사용하지만 자주 변경되지는 않는
// 표준 시스템 포함 파일 또는 프로젝트 관련 포함 파일이
// 들어 있는 포함 파일입니다.
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>



// TODO: 프로그램에 필요한 추가 헤더는 여기에서 참조합니다.
#include <Windows.h>

#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2\highgui.hpp"
#include "opencv2/video/video.hpp"		// for meanshift
#include "opencv2/videoio/videoio.hpp"	

#include <vector>
#include <ppl.h>

#include <boost\foreach.hpp>
#include <boost\tokenizer.hpp>
#include <boost\lexical_cast.hpp>
#include <boost\format.hpp>
#include <boost\filesystem.hpp>
#include <iostream>

#include "GMMA_tracker.h"