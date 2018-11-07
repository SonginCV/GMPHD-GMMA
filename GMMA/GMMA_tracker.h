/*
BSD 2-Clause License

Copyright (c) 2018, Young-min Song,
Machine Learning and Vision Lab(https://sites.google.com/view/mlv/),
Gwangju Institute of Science and Technology(GIST), South Korea.
All rights reserved.

This software is an implementation of the GMPHD filter based tracker.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
this list of conditions and the following disclaimer in the documentation
and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#pragma once

using namespace cv;

#include <iostream>
#include <algorithm>
#include <vector>
#include <list>
#include <map>
#include <ppl.h>
#include <boost/format.hpp>

#include <numeric>
#include <functional>

#include "HungarianAlgorithm.h"

#if CV_MAJOR_VERSION == 3
	#include "opencv2\core\core.hpp"
	#include "opencv2\highgui\highgui.hpp"
	#include "opencv2\imgproc\imgproc.hpp"
	#include "opencv2\opencv.hpp"
#endif

#define PI						3.14159265
#define e						2.71828182
#define DIMS_STATE				6
#define DIMS_STATE_MID			4
#define DIMS_OBSERVATION		4
#define DIMS_OBSERVATION_MID	2
#define PREDICTION_LEVEL_LOW	1
#define PREDICTION_LEVEL_MID	2
#define MAX_OBJECTS				27		// 2^5 - 1, prime number is good to be not duplicated 
#define MAX_GAUSSIANS			128		// 2^7
#define PIXEL_INTERVAL			(10)
#define T_th					(0.0)
#define W_th					(0.0)
#define TRACK_MIN_LENGHT		3		// Tracklet Minimum Length for reliability
//#define TRACK_DA_MIN_LENGTH	3		// Tracklet Minimum Leagth for Data Asssociation (L2S)
#define TRACK_ASSOCIATION_TERM	100		// about 3~4 seconds
#define TRACK_CLEARING_TERM		TRACK_ASSOCIATION_TERM	// Only Dead Trackelets are reset(removed)
#define FRAMES_BATCH_SIZE		1024
#define PROCESSING_TIME_DELAY	50
#define FRAMES_DELAY			(TRACK_MIN_LENGHT-1)
#define FRAME_OFFSET			1				// First frame number of the MOT challenge dataset (1), others(0)
#define LOW_ASSOCIATION_DIMS	2
#define LOW_ASSOCIATION_DIMS_RB	4
#define Q_TH_LOW				0.00000000001	// 속도 제대로 반영후 10씩 더 곱해줌
#define Q_TH_LOW_RB				0.00000001
#define Q_TH_MID				0.00000001
#define P_SURVIVE_LOW			0.99			// object number >=2 : 0.99, else 0.95
#define P_SURVIVE_MID			0.95
#define N_CAMERA				1
#define OBJECT_TYPE_PERSON		1
#define OBJECT_TYPE_CAR			2
#define OBJECT_TYPE_BICYCLE		3
#define OBJECT_TYPE_SUITCASE	4
#define OBJECT_TYPE_CHAIR		5
#define OBJECT_TYPE_TRUCK		6
#define VAR_X					25// 100//25 // 100	// 25
#define VAR_Y					100//400//100//400 // 100 
#define VAR_X_VEL				25//100//25// 100 // 25
#define VAR_Y_VEL				100//400//100//400	// 100
#define VAR_WIDTH				400//100//400 // 100ggg
#define VAR_HEIGHT				900//400//900	//400
#define VAR_X_MID				100// 100//25 // 100	// 25
#define VAR_Y_MID				400//400//100//400 // 100 
#define VAR_X_VEL_MID			100///100//25// 100 // 25
#define VAR_Y_VEL_MID			400//400//100//400	// 100
#define WAIT_FLAG				2
#define UPDATED_FLAG			2
#define OBJ_REGULAR_WIDTH		60
#define OBJ_REGULAR_HEIGHT		120
#define OBJ_RECULAR_SIZE		(OBJ_REGULAR_WIDTH*OBJ_REGULAR_HEIGHT)
#define OBJ_MIN_SIZE			100
#define OBJ_RECULAR_SIZE		(OBJ_REGULAR_WIDTH*OBJ_REGULAR_HEIGHT)

#define DA_LONG2SHORT_USE		1	// 1: use, 0:not use
#define RB_ASSOCIATION_USE		0	// Roll Back Association
#define DB_TYPE_MOT_CHALLENGE	0	// MOT Challenge Dataset
#define DB_TYPE_UA_DETRAC		1   // UA-DETRAC Dataset
#define DB_TYPE_ICT1			2	// ICT 1세부 Dataset
#define DB_TYPE_ICT2			3	// ICT 2세부 Dataset		
#define VELOCITY_UPDATE_ALPHA	0.9f
#define SIZE_UPDATE_BETA		0.9f

#define ASSOCIATION_STAGE_1_ON				1
#define ONLY_SHORT_TRACKLET_ELIMINZATION_ON	0
#define ASSOCIATION_STAGE_2_ON				1
#define OCC_HANDLING_FRAME_WISE_ON			0	// 1: use, 0:not use
#define OCC_HANDLING_TRACKLET_WISE_ON		1	// 1: use, 0:not use
#define GROUP_UNIFYING_MANAGEMENT_ON		1	// For detecting full occlusion
#define MERGE_ON							1
#define MERGE_METRIC_SIOA					1	// 0: use IOU, 1: use SIOA
#define TRACK_APPROXIMIATION_ON				0

#define STAGE_1_VISUALIZATION_ON			0
#define MOTION_PREDICTION_VISUALIZATION		0
#define OCCLUSION_DETECTION_VISUALIZATION	0

#define DEBUG_PRINT			0
#define DEBUG_PRINT_COST	0
#define LOG_PRINT			0
#define LOG_PRINT_COST		0
#define DEBUG_PRINT_MERGE	0
// ICT 1 세부
// 지금까진 전부 400일 제일 좋았다.
// 400 400 400 400 900 900 으로 바꿔보자 400 일 때랑 거의 유사하다.
// 차일때와 사람일때 parameter를 바꿔서 줄 수 있도록 구조를 바꿔야 한다.
// 차의경우 카메라에 진입할때/멀어질때/나갈때 갑자기 크기가 변하는 문제 (위치를 좀더 영향력있게?)
// 사람의 경우 속도가 갑자기 어긋날때 날아가서 id가 바뀌는 문제 (0.4, 0.6 문제로 바꿀수있을듯)

typedef struct boundingbox_id{
	int nid;
	int min_id;
	cv::Rect rec ; // t, t-1, t-2
	boundingbox_id(){
	}
	boundingbox_id(int id, cv::Rect occRect = cv::Rect()) :nid(id){
		// Deep Copy
		nid = id;
		rec = occRect;
	}
	boundingbox_id& operator=(const boundingbox_id& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->nid = copy.nid;
		this->min_id = copy.min_id;
		this->rec = copy.rec;
	}
}RectID;

typedef struct bbTrack {
	int fn;
	int id;
	int id_associated; // it is able to be used in Tracklet-wise association
	cv::Rect rec, rec_t_1, rec_t_2;
	cv::Rect rec_corr;
	float vx, vx_prev;
	float vy, vy_prev;
	float weight;
	cv::Mat cov;
	cv::Mat tmpl;
	cv::Mat hist;
	float density;
	bool isAlive;
	bool isMerged = false;
	bool isLBA = false;
	bool isOcc = false;
	vector<RectID> occTargets;
	bbTrack() {}
	bbTrack(int fn, int id, int isOcc, cv::Rect rec, cv::Mat obj = cv::Mat(), cv::Mat hist = cv::Mat()) :
		fn(fn), id(id), isOcc(isOcc), rec(rec) {
		if (!hist.empty()) {
			this->hist.release();
			this->hist = hist.clone(); // deep copy
		}
		else {
			//this->hist.release();
			//printf("[ERROR]target_bb's parameter \"hist\" is empty!\n");
			this->hist = hist;
		}
		if (!obj.empty()) {
			this->tmpl.release();
			this->tmpl = obj.clone(); // deep copy
		}
		else {
			//this->obj_tmpl.release();
			this->tmpl = obj;
		}
		//isOccCorrNeeded = false; // default
	}
	bool operator<(const bbTrack& trk) const {
		return (id < trk.id);
	}
	bbTrack& operator=(const bbTrack& copy) { // overloading the operator = for deep copy
		if (this == &copy) // if same instance (mermory address)
			return *this;

		this->fn = copy.fn;
		this->id = copy.id;
		this->rec = copy.rec;
		this->vx = copy.vx;
		this->vy = copy.vy;
		this->rec_t_1 = copy.rec_t_1;
		this->rec_t_2 = copy.rec_t_2;
		this->rec_corr = copy.rec_corr;
		this->vx_prev = copy.vx_prev;
		this->vy_prev = copy.vy_prev;
		this->density = copy.density;
		this->isAlive = copy.isAlive;
		this->isMerged = copy.isMerged;
		this->isLBA = copy.isLBA;
		this->isOcc = copy.isOcc;
		this->weight = copy.weight;

		if (!cov.empty()) this->cov = copy.cov.clone();
		if (!tmpl.empty()) this->tmpl = copy.tmpl.clone();
		if (!hist.empty()) this->hist = copy.hist.clone();

		return *this;
	}
	void CopyTo(bbTrack& dst) {
		dst.fn = this->fn;
		dst.id = this->id;
		dst.rec = this->rec;
		dst.vx = this->vx;
		dst.vy = this->vy;
		dst.rec_t_1 = this->rec_t_1;
		dst.rec_t_2 = this->rec_t_2;
		dst.rec_corr = this->rec_corr;
		dst.vx_prev = this->vx_prev;
		dst.vy_prev = this->vy_prev;
		dst.density = this->density;
		dst.isAlive = this->isAlive;
		dst.isMerged = this->isMerged;
		dst.isLBA = this->isLBA;
		dst.isOcc = this->isOcc;
		dst.weight = this->weight;

		if (!this->cov.empty()) dst.cov = this->cov.clone();
		if (!this->tmpl.empty()) dst.tmpl = this->tmpl.clone();
		if (!this->hist.empty()) dst.hist = this->hist.clone();
	}
	void Destroy() {
		if (!this->cov.empty()) this->cov.release();
		if (!this->tmpl.empty()) this->tmpl.release();
		if (!this->hist.empty()) this->hist.release();
	}

}BBTrk;
typedef struct bbDet {
	int fn;
	cv::Rect rec;
	float confidence;
	float weight; // normalization value of confidence at time t
	int id;// Used in Looking Back Association
}BBDet;


class SYM_MOT_HGMPHD{

public:
	SYM_MOT_HGMPHD();
	SYM_MOT_HGMPHD(int objType, int width, int height);
	~SYM_MOT_HGMPHD();
	double* DoMOT(int iFrmCnt, const cv::Mat& img, int& nTargets, double *bbsDet, int nBbsDet, double Th_Conf = 0.0);

	// Set the number of total frames
	void SetTotalFrames(int num_total_frames) {
		this->iTotalFrames = num_total_frames; 
		if (DEBUG_PRINT) {
			printf("Total Processing Frames:%d\n", this->iTotalFrames);
			//cv::waitKey();
		}
	}
	void Destroy();
private:
	cv::Mat F;		// transition matrix state_t-1 to state_t 	
	cv::Mat Q;		// process noise covariance
	cv::Mat Ps;		// covariance of states's Gaussian Mixtures for Survival
	cv::Mat R;		// the covariance matrix of measurement
	cv::Mat H;		// transition matrix state_t to observation_t

	cv::Mat F_mid;	// transition matrix state_t-1 to state_t 	
	cv::Mat Q_mid;	// process noise covariance
	cv::Mat Ps_mid;	// covariance of states's Gaussian Mixtures for Survival
	cv::Mat R_mid;	// the covariance matrix of measurement
	cv::Mat H_mid;	// transition matrix state_t to observation_t


	bool isInitialization;
	
	// Private Members
	static int iTrackerCnt;
	int iTrackerID;

	double P_survive = P_SURVIVE_LOW;		// Probability of Survival	(User Parameter)(Constant)
	double P_survive_mid = P_SURVIVE_MID;
	
	int frmWidth;
	int frmHeight;

	int iTotalFrames=-1;
private:

	std::vector<BBTrk> liveTrkVec;							// tracking states at now
	std::vector<BBTrk> liveTracksBatch[TRACK_MIN_LENGHT];
	std::vector<BBTrk> lostTrkVec;							// tracking states at now
	std::vector<BBTrk> lostTracksBatch[TRACK_MIN_LENGHT];

	std::map<int, vector<RectID>> groupsBatch[3];					// THe container for Group management, index (0:t-d-2, 1:t-d-1, 2:t-d), d: delayed time

	std::map<int, std::vector<BBTrk>> tracksbyID;
	std::map<int, std::vector<BBTrk>> tracks_reliable;
	std::map<int, std::vector<BBTrk>> tracks_unreliable;
	//std::map<int, std::vector<BBTrk>> tracks_live; // not used
	//std::map<int, std::vector<BBTrk>> tracks_lost; // not used


	int sysFrmCnt;
	int usedIDcnt;

	FILE *g_fp = NULL;
	fpos_t g_fpos = 0;

	string seqName;
	string detName;

	int noDetectionPeriod;
public:
	std::vector<std::vector<BBTrk>> allLiveReliables; // for UA-DETRAC requiring such a idiot format.
public:
	cv::Mat imgBatch[TRACK_MIN_LENGHT];
	std::vector<BBDet> detsBatch[TRACK_MIN_LENGHT];

	CvScalar color_tab[MAX_OBJECTS];
	int db_type;
	int trackObjType;
	int cam_num = 0;
private:
	void InitializeImagesQueue(int width, int height);
	void InitializeColorTab();
	void InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, int dims_state, int dims_obs);

	float FrameWiseAffinity(BBDet ob, BBTrk& stat_temp, const int dims = 2);
	float TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int& dims=2);
	float TrackletWiseAffinityVelocity(BBTrk &stat_pred, const BBTrk& obs, const int& dims=4);


	double GaussianFunc(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat);

	double ApproximateGaussianProbability(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat);
	// Prediction of state_k|k-1 from state_k-1 (x,y,vel_x,vel_y,width, height) using Kalman filter
	void PredictFrmWise(int iFrmCnt, vector<BBTrk>& stats, const cv::Mat F, const cv::Mat Q, cv::Mat &Ps, int iPredictionLevel);
	void QuadraticMotionEstimation(const vector<BBTrk>& stats); // Linear 와 다르게 prediction point 를 매번주면 비용이 증가하므로,, association 성공 할때만 하고 parameter 저장해두고 있어야 한다.
	cv::Point2f LinearMotionEstimation(map<int, vector<BBTrk>> tracks, int id);
	cv::Point2f LinearMotionEstimation(vector<BBTrk> tracklet);

	// Tracklets Management
	void ArrangeTargetsVecsBatchesLiveLost();
	void PushTargetsVecs2BatchesLiveLost();
	void SortTrackletsbyID(map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& targets);
	void ClassifyTrackletReliability(int iFrmCnt, map<int, vector<BBTrk>>& tracksbyID, map<int, vector<BBTrk>>& reliables, map<int, std::vector<BBTrk>>& unreliables);
	void ClassifyReliableTracklets2LiveLost(int iFrmCnt, const map<int, vector<BBTrk>>& reliables, vector<BBTrk>& liveReliables, vector<BBTrk>& LostReliables);
	void ArrangeRevivedTracklets(map<int, vector<BBTrk>>& tracks, vector<BBTrk>& lives);
	void CheckOcclusionsMergeStates(vector<BBTrk>& stats, const double T_merge=1.0, const double T_occ=0.0);
	void CheckOcclusionsGroups(vector<BBTrk>& stats, const double T_merge = 2.0, const double T_occ = 0.5);
	void ClearOldEmptyTracklet(int current_fn, map<int, vector<BBTrk>>& tracklets, int MAXIMUM_OLD= TRACK_CLEARING_TERM);
	// Group Management
	void UnifyNeighborGroups(vector<BBTrk> input_targets); 
	int  FindMinIDofNeigborsRecursive(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id);	// Recursive function
	int  FindMinIDofNeigbors2Depth(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id);		// Only 2 depth search


	// Return Tracking Results
	double* ReturnTrackingResults(int iFrmCnt, const vector<BBTrk>& results, int& nTargets, int rows=5);
	double* ReturnTrackingResults(int iFrmCnt, const vector<BBTrk>& allTargets, const vector<BBTrk>& reliables, int& nTargets, int rows=5);

	// Calculate the minimum cost pairs by Hungarian method (locally optimized)
	std::vector<vector<int>> HungarianMethod(int* r, int nObs, int nStats, int min_cost=0);
	std::vector<vector<int>> HungarianMethod(double* r, int nObs, int nStats, double min_cost = 0);
	std::vector<std::vector<double> > array_to_matrix_dbl(int* m, int rows, int cols, int min_cost = 0);
	std::vector<std::vector<double> > array_to_matrix_dbl(double* m, int rows, int cols, double min_cost = 0);
	
	// Find the min cost pairs by Greedy algorithm (not optimized)
	std::vector<std::vector<int>> GreedyAssignMinCostPairs(int* r, int nObs, int nStatss);
	std::vector<std::vector<int> > array_to_matrix_int(int* m, int rows, int cols, bool SQUARE_MATRIX = false);

	bool IsOutOfFrame(int x, int y, int width, int height, int fWidth, int fHeight);

	// Data association including update & pruning
	void DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, vector<BBDet>& obss, cv::Mat &Ps, const cv::Mat& H, double P_survive = 0.99, int offset = FRAME_OFFSET, int dims_low = LOW_ASSOCIATION_DIMS);
	void DataAssocTrkWise(int iFrmCnt, cv::Mat& img, vector<BBTrk>& stats_lost, vector<BBTrk>& obss_live);
	
	// Rect Region Correction for preventing out of frame before object region cropping
	cv::Rect RectExceptionHandling(int fWidth, int fHeight, cv::Rect rect);
	
	// Object Region Cropping
	void CropObjRegionRegularization(IplImage* frame_input, cv::Rect rec, cv::Mat &obj_tmpl, bool regularSize=true);
	void CropRegularSizeObj(const cv::Mat& img, cv::Rect rec, cv::Mat &obj_tmpl, bool regularSize = true);
	
	// Template Matching object by object
	double MatchingMethod(cv::Mat input_templ, cv::Mat cand, int cand_index = 0, int match_method = CV_TM_SQDIFF);
	
	// Local Search Function
	void GetObjectHistogram(Mat &frame, Mat objectTmpl, Rect objectRegion, Mat& objectHist, Rect& LocalSearchRegion);
	void BackProjection(const Mat &frame, const Mat &obj_hist, Mat &bp);

	// Sort tracklet vector by ID (ascending order)
	vector<BBTrk> SortTargetsbyID(double* bbs_tracks);

	// Empty Tracklets Clear

	// Old Tracklets Clear for Memory Management

	// Write Tracking Results in file
	void WriteTrackingResults(int cam_num, int iFrame_count, int Nbbs, const double *bbs, const int column_elements=5, int sync_offset = FRAME_OFFSET, string dbname = "", string tag = "", double ThDetConf=0.0);

	// Read Detection Results from txt file

	// Util
	void SaveAImage(int iFrame_count, const CvArr* image, string tag, string ext) {

		// Make an image file name 
		string strFilePath = tag;

		char fileName[16];
		sprintf(fileName, "%.7d", iFrame_count);
		strFilePath = strFilePath + string(fileName)+ + "." + ext;

		cvSaveImage(strFilePath.c_str(), image);
	}
	void PrintCostMatrix(int *C, int nStats, int mObss, int max_cost);
	void PrintCostMatrix(double *C, int nStats, int mObss, double max_cost);

	cv::Mat detailViewImg;
public:
	void SetSeqName(string seq);
	string GetSeqName();
	void SetDetName(string det);
	string GetDetName();
	// Check occlusion between object region and use setting occlusion rection
	bool IsOccluded(int &occlusionCase, CvRect obj1, CvRect obj2, CvRect &occ = cvRect(0, 0, 1, 1));
	void cvBoundingBox(IplImage* img, CvPoint lt, CvPoint rb, CvScalar color = cvScalar(0, 0, 255), int thick=2,int id = -1, string type="");
	void cvBoundingBox(cv::Mat& img, cv::Rect rec, cv::Scalar color = cvScalar(0, 0, 255), int thick = 2, int id = -1, string type = "");
	void cvPrintMat(cv::Mat matrix, string name = "");
};