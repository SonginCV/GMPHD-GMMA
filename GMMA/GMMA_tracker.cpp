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

#include "stdafx.h"
#include "GMMA_tracker.h"

//using namespace cv;

int SYM_MOT_HGMPHD::iTrackerCnt = 0;

SYM_MOT_HGMPHD::SYM_MOT_HGMPHD() {
	this->iTrackerID = ++SYM_MOT_HGMPHD::iTrackerCnt;
	this->usedIDcnt = 0;
	this->noDetectionPeriod = 0;
	/*for (int i = 0; i < TRACK_MIN_LENGHT; i++)
		frames_queue[i] = cvCreateImage(cvSize(640, 480), IPL_DEPTH_8U, 3);*/

	this->frmWidth = -1;
	this->frmHeight = -1;

	isInitialization = false;
	this->sysFrmCnt = 0;

	InitializeColorTab();

	InitializeMatrices(F, Q, Ps, R, H, DIMS_STATE, DIMS_OBSERVATION);
	InitializeMatrices(F_mid, Q_mid, Ps_mid, R_mid, H_mid, DIMS_STATE_MID, DIMS_OBSERVATION_MID);
}
SYM_MOT_HGMPHD::SYM_MOT_HGMPHD(int objType, int width, int height) {
	this->iTrackerID = ++SYM_MOT_HGMPHD::iTrackerCnt;
	this->usedIDcnt = 0;
	this->trackObjType = objType;
	this->noDetectionPeriod = 0;

	this->frmWidth = width;
	this->frmHeight = height;

	isInitialization = false;
	this->sysFrmCnt = 0;

	InitializeImagesQueue(width, height);

	InitializeColorTab();

	InitializeMatrices(F, Q, Ps, R, H, DIMS_STATE, DIMS_OBSERVATION);
	InitializeMatrices(F_mid, Q_mid, Ps_mid, R_mid, H_mid, DIMS_STATE_MID, DIMS_OBSERVATION_MID);
}
void SYM_MOT_HGMPHD::InitializeImagesQueue(int width, int height) {
	for (int i = 0; i < TRACK_MIN_LENGHT; i++)
		this->imgBatch[i] = cv::Mat(height, width, CV_8UC3);
	this->frmWidth = width;
	this->frmHeight = height;
}
SYM_MOT_HGMPHD::~SYM_MOT_HGMPHD() {

}
void SYM_MOT_HGMPHD::Destroy() {

	--SYM_MOT_HGMPHD::iTrackerCnt;


	F.release();
	Q.release();
	Ps.release();
	R.release();
	H.release();

	F_mid.release();
	Q_mid.release();
	Ps_mid.release();
	R_mid.release();
	H_mid.release();

	for (int i = 0; i < TRACK_MIN_LENGHT; i++) {
		imgBatch[i].release();
		liveTracksBatch[i].clear();
		lostTracksBatch[i].clear();
		detsBatch[i].clear();
	}

	// Clearing Gaussian Mixture Containers
	vector<BBTrk>::iterator iterT;
	for (iterT = liveTrkVec.begin(); iterT != liveTrkVec.end(); ++iterT)
	{
		iterT->occTargets.clear();
		iterT->Destroy();
	}
	liveTrkVec.clear();
	for (iterT = lostTrkVec.begin(); iterT != lostTrkVec.end(); ++iterT)
	{
		iterT->occTargets.clear();
		iterT->Destroy();
	}
	lostTrkVec.clear();

	cvDestroyAllWindows();

	// Clearing Tracklets Containers
	

	// Draw reliable tracklets, 프레임 구간 별로 나눠서 Tracklet Visualization
	//cvDrawTracklets(tracklets_reliable, g_inputImgPaths, g_iFrameCount, color_tab);	
	//cvDrawTracklets(tracklets_ed_reliable, g_inputImgPaths, g_iFrameCount, color_tab, "ed");
	//cvDrawTracklets(tracklets_ing_reliable, g_inputImgPaths, g_iFrameCount, color_tab, "ing");

}
// Initialize Color Tab
void SYM_MOT_HGMPHD::InitializeColorTab()
{
	int a;
	for (a = 1; a*a*a < MAX_OBJECTS; a++);
	int n = 255 / (a - 1);
	IplImage *temp = cvCreateImage(cvSize(40 * (MAX_OBJECTS), 32), IPL_DEPTH_8U, 3);
	cvSet(temp, CV_RGB(0, 0, 0));
	for (int i = 0; i < a; i++) {
		for (int j = 0; j < a; j++) {
			for (int k = 0; k < a; k++) {
				//if(i*a*a+j*a+k>MAX_OBJECTS) break;
				//printf("%d:(%d,%d,%d)\n",i*a*a +j*a+k,i*n,j*n,k*n);
				if (i*a*a + j*a + k == MAX_OBJECTS) break;
				color_tab[i*a*a + j*a + k] = CV_RGB(i*n, j*n, k*n);
				cvLine(temp, cvPoint((i*a*a + j*a + k) * 40 + 20, 0), cvPoint((i*a*a + j*a + k) * 40 + 20, 32), CV_RGB(i*n, j*n, k*n), 32);
			}
		}
	}
	//cvShowImage("(private)Color tap", temp);
	cvWaitKey(1);
	cvReleaseImage(&temp);
}
void SYM_MOT_HGMPHD::InitializeMatrices(cv::Mat &F, cv::Mat &Q, cv::Mat &Ps, cv::Mat &R, cv::Mat &H, int dims_state, int dims_obs)
{
	/* Initialize the transition matrix F, from state_t-1 to state_t

	1	0  △t	0	0	0
	0	1	0  △t	0	0
	0	0	1	0	0	0
	0	0	0	1	0	0
	0	0	0	0	1	0
	0	0	0	0	0	1

	△t = 구현시에는 △frame으로 즉 1이다.
	*/
	F = cv::Mat::eye(dims_state, dims_state, CV_64FC1); // identity matrix
	F.at<double>(0, 2) = 1.0;///30.0; // 30fps라 가정, 나중에 계산할때 St = St-1 + Vt-1△t (S : location) 에서 
	F.at<double>(1, 3) = 1.0;///30.0; // Vt-1△t 의해 1/30 은 사라진다. Vt-1 (1frame당 이동픽셀 / 0.0333..), △t = 0.0333...

	if (dims_state == DIMS_STATE) {
		Q = (Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0.0, 0, 0, 0, \
			0, VAR_Y, 0, 0.0, 0, 0, \
			0.0, 0, VAR_X_VEL, 0, 0, 0, \
			0, 0.0, 0, VAR_Y_VEL, 0, 0, \
			0, 0, 0, 0, 0, 0, \
			0, 0, 0, 0, 0, 0);
		Q = 0.5 * Q;

		Ps = (Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, 0, 0, \
			0, VAR_Y, 0, 0, 0, 0, \
			0, 0, VAR_X_VEL, 0, 0, 0, \
			0, 0, 0, VAR_Y_VEL, 0, 0, \
			0, 0, 0, 0, VAR_WIDTH, 0, \
			0, 0, 0, 0, 0, VAR_HEIGHT);

		R = (Mat_<double>(dims_obs, dims_obs) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);
		/*	Initialize the transition matrix H, transing the state_t to the observation_t(measurement) */
		H = (Mat_<double>(dims_obs, dims_state) << \
			1, 0, 0, 0, 0, 0, \
			0, 1, 0, 0, 0, 0, \
			0, 0, 0, 0, 1, 0, \
			0, 0, 0, 0, 0, 1);
	}
	else if (dims_state == DIMS_STATE_MID) {
		Q = (Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);
		Q = 0.5 * Q;

		Ps = (Mat_<double>(dims_state, dims_state) << \
			VAR_X, 0, 0, 0, \
			0, VAR_Y, 0, 0, \
			0, 0, VAR_X_VEL, 0, \
			0, 0, 0, VAR_Y_VEL);

		R = (Mat_<double>(dims_obs, dims_obs) << \
			VAR_X, 0, \
			0, VAR_Y);

		/*	Initialize the transition matrix H, transing the state_t to the observation_t(measurement) */
		H = (Mat_<double>(dims_obs, dims_state) << \
			1, 0, 0, 0, \
			0, 1, 0, 0);
	}
}
double SYM_MOT_HGMPHD::ApproximateGaussianProbability(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat)
{
	double probability = -1.0;
	if ((x.rows != m.rows) || (cov_mat.rows != cov_mat.cols) || (x.rows != D)) {
		printf("[ERROR](x.rows!=m.rows) || (cov_mat.rows!=cov_mat.cols) || (x.rows!=D) (line:258)\n");
	}
	else {
		cv::Mat sub(D, 1, CV_64FC1);
		cv::Mat power(1, 1, CV_64FC1);
		double exponent = 0.0;
		double coefficient = 1.0;

		sub = x - m;
		power = sub.t() * cov_mat.inv(DECOMP_SVD) * sub;

		coefficient = ((1.0) / (pow(2.0*PI, (double)D / 2.0)*pow(cv::determinant(cov_mat), 0.5)));
		exponent = (-0.5)*(power.at<double>(0, 0));
		probability = coefficient*pow(e, exponent);

		sub.release();
		power.release();
	}
	return probability;
}
double SYM_MOT_HGMPHD::GaussianFunc(int D, cv::Mat x, cv::Mat m, cv::Mat cov_mat) {
	double probability = -1.0;
	if ((x.rows != m.rows) || (cov_mat.rows != cov_mat.cols) || (x.rows != D)) {
		printf("[ERROR](x.rows!=m.rows) || (cov_mat.rows!=cov_mat.cols) || (x.rows!=D) (line:258)\n");
	}
	else {
		cv::Mat sub(D, 1, CV_64FC1);
		cv::Mat power(1, 1, CV_64FC1);
		double exponent = 0.0;
		double coefficient = 1.0;

		sub = x - m;
		power = sub.t() * cov_mat.inv(cv::DECOMP_SVD) * sub;

		coefficient = ((1.0) / (pow(2.0*PI, (double)D / 2.0)*pow(cv::determinant(cov_mat), 0.5)));
		exponent = (-0.5)*(power.at<double>(0, 0));
		probability = coefficient*pow(e, exponent);

		sub.release();
		power.release();
	}
	if (probability < FLT_MIN) probability = 0.0;

	//if (0) { // GpuMat
	//	cv::cuda::GpuMat subGpu;
	//	cv::cuda::GpuMat powerGpu;
	//}
	return probability;
}
double* SYM_MOT_HGMPHD::ReturnTrackingResults(int iFrmCnt, const vector<BBTrk>& results, int& nTargets, int rows) {

	double *bbs_track = (double *)malloc(results.size() * rows * sizeof(double));
	int nStats = 0;
	vector<BBTrk>::const_iterator iterT;
	for (iterT = results.begin(); iterT != results.end(); ++iterT) {

		if (iterT->isAlive == true) {

			// Copy the tracking results for writing them in file
			bbs_track[nStats * 5 + 0] = iterT->id;//+6, +2, +28, +13, +0(no error), 
			bbs_track[nStats * 5 + 1] = iterT->rec.x;
			bbs_track[nStats * 5 + 2] = iterT->rec.y;
			bbs_track[nStats * 5 + 3] = iterT->rec.width;
			bbs_track[nStats * 5 + 4] = iterT->rec.height;
			nStats++;

			// Display Tracked Target Bounding Boxes with ID(Not delayed and pruned)
			//cvBoundingBox(img_trk, cvPoint(x, y), cvPoint(x + width, y + height), color_tab[iterG->id % (MAX_OBJECTS - 1)], 2,iterG->id);
			//if (DEBUG_PRINT)
				//printf("[%d]%d:(%d,%d,%d,%d,%lf,%lf,%lf)\n", iFrmCnt, iterT->id, iterT->rec.x, iterT->rec.y, iterT->rec.width, iterT->rec.height, iterT->vx, iterT->vy, iterT->weight);
		}
	}

	nTargets = nStats;

	if (DEBUG_PRINT) printf("# of Objects: %d\n", nStats);
	return bbs_track;
}
double* SYM_MOT_HGMPHD::ReturnTrackingResults(int iFrmCnt, const vector<BBTrk>& allTargets, const vector<BBTrk>& reliables, int& nTargets, int rows) {
	double *bbs_all_tracks = (double *)malloc(allTargets.size() * rows * sizeof(double));
	int nStats = 0;
	vector<BBTrk>::const_iterator iterT;
	for (iterT = allTargets.begin(); iterT != allTargets.end(); ++iterT) {

		bool isReliable = false;
		vector<BBTrk>::const_iterator iterTR;
		for (iterTR = reliables.begin(); iterTR != reliables.end(); ++iterTR) {
			if (iterT->id == iterTR->id) {
				isReliable = true;
				break;
			}
		}
		// Select the targets which exists reliable targets vector.
		if (iterT->isAlive == true && isReliable) {

			// Copy the tracking results for writing them in file
			bbs_all_tracks[nStats * 5 + 0] = iterT->id;//+6, +2, +28, +13, +0(no error), 
			bbs_all_tracks[nStats * 5 + 1] = iterT->rec.x;
			bbs_all_tracks[nStats * 5 + 2] = iterT->rec.y;
			bbs_all_tracks[nStats * 5 + 3] = iterT->rec.width;
			bbs_all_tracks[nStats * 5 + 4] = iterT->rec.height;
			nStats++;
		}
	}
	double *bbs_reliable_tracks = (double *)malloc(nStats * rows * sizeof(double));
	for (int i = 0; i < nStats; i++) {
		bbs_reliable_tracks[i * 5 + 0] = bbs_all_tracks[i * 5 + 0];
		bbs_reliable_tracks[i * 5 + 1] = bbs_all_tracks[i * 5 + 1];
		bbs_reliable_tracks[i * 5 + 2] = bbs_all_tracks[i * 5 + 2];
		bbs_reliable_tracks[i * 5 + 3] = bbs_all_tracks[i * 5 + 3];
		bbs_reliable_tracks[i * 5 + 4] = bbs_all_tracks[i * 5 + 4];
	}

	nTargets = nStats;
	free(bbs_all_tracks);
	if (DEBUG_PRINT) printf("# of Objects: %d\n", nStats);
	return bbs_reliable_tracks;
}
// System에서 주는 iFrameCnt의 처음 값이 0이 아닐수 있으므로
// 내부적으로 DoLocalTracking이 불리는 횟수를 세는 gFrameCnt를 따로 둔다.
// line: 334-2474
double* SYM_MOT_HGMPHD::DoMOT(int iFrmCnt, const cv::Mat& img, int& nTargets, double *bbsDet, int nBbsDet, double Th_Conf) {
	// https://kr.mathworks.com/help/vision/examples/motion-based-multiple-object-tracking.html?requestedDomain=www.mathworks.com#zmw57dd0e2778
	// 위에거 참고해서 프레임워크 정리해보자
	if (DEBUG_PRINT) printf("\n");
	//this->sysFrmCnt = iFrmCnt;

	if (this->sysFrmCnt == 0)
		InitializeImagesQueue(img.cols, img.rows);

	// Copy image for drawing procesures in detail.
	//this->detailViewImg = img.clone();

	// Load the Detection Results
	// Detection filtering can be applied here
	std::vector<BBDet> detVec;
	for (int d = 0; d < nBbsDet; ++d) {
		BBDet bbd;
		bbd.fn = iFrmCnt;
		bbd.rec.x = bbsDet[0 * nBbsDet + d];
		bbd.rec.y = bbsDet[1 * nBbsDet + d];
		bbd.rec.width = bbsDet[2 * nBbsDet + d];
		bbd.rec.height = bbsDet[3 * nBbsDet + d];
		bbd.confidence = bbsDet[4 * nBbsDet + d];
		//if (DEBUG_PRINT) printf("[%d](%d,%d,%d,%d,%lf)\n",iFrmCnt,bbd.rec.x,bbd.rec.y,bbd.rec.width,bbd.rec.height,bbd.confidence);
		if (bbd.confidence >= Th_Conf) detVec.push_back(bbd);
	}
	if (DEBUG_PRINT) cout << "(1)";
	// Normalization
	vector<BBDet>::iterator iterDet;
	double sumConf = 0.0;
	for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
		sumConf += iterDet->confidence;
	}
	if (sumConf > 0.0) {
		for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
			iterDet->weight = iterDet->confidence / sumConf;
		}
	}
	else if (sumConf == 0.0) {
		for (iterDet = detVec.begin(); iterDet != detVec.end(); iterDet++) {
			iterDet->weight = 0.0;
		}
	}
	if (DEBUG_PRINT) cout << "(2)";

	// Keep the images and observations into vector array within the recent 10 frames 
	if (this->sysFrmCnt >= TRACK_MIN_LENGHT) {

		for (int q = 0; q < FRAMES_DELAY; q++) {
			imgBatch[q + 1].copyTo(imgBatch[q]);
		}
		img.copyTo(imgBatch[FRAMES_DELAY]);
		detsBatch[FRAMES_DELAY] = detVec;
	}
	else if (this->sysFrmCnt < TRACK_MIN_LENGHT) {
		img.copyTo(imgBatch[this->sysFrmCnt]);
		detsBatch[this->sysFrmCnt] = detVec;
	}
	if (DEBUG_PRINT) cout << "(3)";

	if (DEBUG_PRINT) printf("[%d] %d Observations\n", this->sysFrmCnt, detVec.size());


	if (!this->liveTrkVec.empty()) { // 이게 if 문에서 먼저와야 한다.
		//if (DEBUG_PRINT) cout << "(b)";
		/* DoPrediction
		Step 1: Prediction (low level)
		Assumptions
		[1] Prediction is executed from 2nd frame
		[2] The velocities in x & y-axis at the initial frame are all zeros.
		[3] The state k|k-1 at k-th frame are predicted by the state k-1(the poisition and velocity at the k-1 th frames)
		*/
		//DoPrediction(GMM, F, Q, Ps, iFrameCnt, iObjCur, obs);
		//printf("[%d]Prediction\n", this->sysFrmCnt);
		PredictFrmWise(iFrmCnt, liveTrkVec, F, Q, Ps, PREDICTION_LEVEL_LOW);
		if (DEBUG_PRINT) cout << "(Pred-Stg1)";

		/* DoDataAssociation (low level)
		Step 2. Update after data Association for matching the current observations with the previous targets states
		Step 3. Pruning the gaussians with the weight under threshold (Tth)
		*/
		//DoDataAssociation(frame_track, GMM, obs, iObjCur, iObjPrev, Ps, H, P_survive, iFrameCnt, offset);
		//printf("[%d]DA-low-1\n", this->sysFrmCnt);
		DataAssocFrmWise(iFrmCnt, img, liveTrkVec, detVec, this->Ps, this->H, this->P_survive, FRAME_OFFSET);
		if (DEBUG_PRINT) cout << "(Assoc-Stg1)";
		// 1. # of targets at the previous(k-1) frame before data association
		// 2. # of targets at the current(k) frame after data association


		if (OCC_HANDLING_FRAME_WISE_ON) {

			//this->CheckOcclusionsMergeStates(this->liveTrkVec, 2.0, 0.5); // do only occlusion check
			vector<BBTrk>::iterator iterR;
			for (iterR = liveTrkVec.begin(); iterR != liveTrkVec.end(); ++iterR) {
				vector<RectID>::iterator iterOCC;
				if (!iterR->occTargets.empty())
				{
					cv::Rect rect_union = iterR->rec;;
					for (iterOCC = iterR->occTargets.begin(); iterOCC != iterR->occTargets.end(); ++iterOCC) {
						rect_union = rect_union | iterOCC->rec;

						// 잘 안되면 tracks_reliable 쓰는 수밖에 근데 이거 key 접근은 속도를 저하시켜서 문제다.
						//cv::Point pt0(iterOCC->occTargetRects[0].x + iterOCC->occTargetRects[0].width / 2.0, iterOCC->occTargetRects[0].y + iterOCC->occTargetRects[0].height / 2.0);
						//cv::Point pt1(iterOCC->occTargetRects[1].x + iterOCC->occTargetRects[1].width / 2.0, iterOCC->occTargetRects[1].y + iterOCC->occTargetRects[1].height / 2.0);

					}
					if (OCCLUSION_DETECTION_VISUALIZATION) {
						cv::Mat img_no_latency = img.clone();

						/*if (this->tracksbyID[iterR->id].size() >= 3) {

							this->cvBoundingBox(img_no_latency, rect_union, cv::Scalar(0, 0, 255), 4);

							cv::Rect rec_corrected = iterR->rec_t_1;
							cv::Point pt_t_2(iterR->rec_t_2.x + iterR->rec_t_2.width / 2.0, iterR->rec_t_2.y + iterR->rec_t_2.height / 2.0);
							cv::Point pt_t_1(iterR->rec_t_1.x + iterR->rec_t_1.width / 2.0, iterR->rec_t_1.y + iterR->rec_t_1.height / 2.0);

							int xc = rec_corrected.x + rec_corrected.width / 2.0;
							int yc = rec_corrected.y + rec_corrected.height / 2.0;
							int vx = pt_t_1.x - pt_t_2.x;
							int vy = pt_t_1.y - pt_t_2.y;

							this->cvBoundingBox(img_no_latency, rec_corrected, this->color_tab[iterR->id % 26], 1, iterR->id);
							cv::arrowedLine(img_no_latency, cv::Point(xc, yc), cv::Point(xc + vx, yc + vy), this->color_tab[iterR->id % 26], 1);

							rec_corrected.x += vx;
							rec_corrected.y += vy;

							this->cvBoundingBox(img_no_latency, rec_corrected, this->color_tab[iterR->id % 26], 2, iterR->id);
						}*/
						
						if (this->tracksbyID[iterR->id].size() >= 3) {
							vector<BBTrk> vecTrk = this->tracksbyID[iterR->id];

							cv::Rect rec_corrected = vecTrk.at(vecTrk.size() - 2).rec;
							cv::Rect rec_prev = vecTrk.at(vecTrk.size() - 3).rec;

							int xc = rec_corrected.x + rec_corrected.width / 2.0;
							int yc = rec_corrected.y + rec_corrected.height / 2.0;

							cv::Point pt_t_2(rec_prev.x + rec_prev.width / 2.0, rec_prev.y + rec_prev.height / 2.0);
							cv::Point pt_t_1(xc, yc);

							int vx = pt_t_1.x - pt_t_2.x;
							int vy = pt_t_1.y - pt_t_2.y;

							this->cvBoundingBox(img_no_latency, rec_corrected, this->color_tab[iterR->id % 26], 1, iterR->id);
							cv::arrowedLine(img_no_latency, cv::Point(xc, yc), cv::Point(xc + vx, yc + vy), this->color_tab[iterR->id % 26], 1);

							rec_corrected.x += vx;
							rec_corrected.y += vy;

							this->cvBoundingBox(img_no_latency, rec_corrected, this->color_tab[iterR->id % 26], 2, iterR->id);

							this->cvBoundingBox(img_no_latency, rect_union, cv::Scalar(0, 0, 255), 4);
						}

						cv::imshow("Occlusion Handling", img_no_latency);
						cv::waitKey(1);
						img_no_latency.release();
					}
				}
			}
		}
	}
	// Step 0: Initializtion
	else if (this->sysFrmCnt == 0 || this->liveTrkVec.size() == 0) {
		// gmm.size()==0 은 첫프레임에 아무런 객체가 없을경우,
		// 객체가 나타나는 최초의 프레임부터 initialize 처리해주기 위함
		//if (DEBUG_PRINT) cout << "(A)";
		std::vector<BBDet>::iterator iterD;
		for (iterD = detVec.begin(); iterD != detVec.end(); ++iterD)
		{
			int id = this->usedIDcnt++;
			/*if (SYM_MOT_HGMPHD::iTrackerCnt == 6) {
			if (this->trackObjType == OBJECT_TYPE_CAR)		id = 6 * id;
			if (this->trackObjType == OBJECT_TYPE_PERSON)	id = 6 * id + 1;
			if (this->trackObjType == OBJECT_TYPE_BICYCLE)	id = 6 * id + 2;
			if (this->trackObjType == OBJECT_TYPE_SUITCASE) id = 6 * id + 3;
			if (this->trackObjType == OBJECT_TYPE_CHAIR)	id = 6 * id + 4;
			if (this->trackObjType == OBJECT_TYPE_TRUCK)	id = 6 * id + 5;
			}
			else if (SYM_MOT_HGMPHD::iTrackerCnt == 1) {
			if (this->trackObjType == OBJECT_TYPE_PERSON)	id = id;
			}
			else if (SYM_MOT_HGMPHD::iTrackerCnt == 2) {
			if (this->trackObjType == OBJECT_TYPE_CAR)		id = 2 * id;
			if (this->trackObjType == OBJECT_TYPE_PERSON)	id = 2 * id + 1;
			}*/

			BBTrk bbt;
			bbt.isAlive = true;
			bbt.id = id;
			bbt.fn = iFrmCnt;
			bbt.rec = iterD->rec;
			bbt.vx = 0.0;
			bbt.vy = 0.0;
			bbt.weight = iterD->weight;

			bbt.cov = (cv::Mat_<double>(4, 4) << \
				VAR_X, 0, 0, 0, \
				0, VAR_Y, 0, 0,
				0, 0, VAR_X_VEL, 0,
				0, 0, 0, VAR_Y_VEL);

			cv::Rect tmplRect;
			tmplRect = RectExceptionHandling(img.cols, img.rows, tmplRect);
			if (tmplRect.width * tmplRect.height >= OBJ_MIN_SIZE) {
				cv::Mat tTmpl;
				CropRegularSizeObj(img, tmplRect, tTmpl);
				bbt.tmpl = tTmpl.clone();
				tTmpl.release();
			}
			this->liveTrkVec.push_back(bbt);
			//this->usedIDcnt++; // global or static variable
		}
	}
	if (DEBUG_PRINT) cout << "(4)";

	// Arrange the targets which have been alive or not (live or lost)
	this->ArrangeTargetsVecsBatchesLiveLost();

	// Push the Tracking Results (live, lost) into the each Tracks Queue (low level)
	/// Keep only the tracking targets at now (except the loss targets)
	this->PushTargetsVecs2BatchesLiveLost();

	if (DEBUG_PRINT) cout << "(5)";
	if (ASSOCIATION_STAGE_1_ON && !ASSOCIATION_STAGE_2_ON) {

		cv::Mat img_trk = img.clone();	// image for no latent association tracking

		detVec.clear();
		img_trk.release();

		double* track_results;

		if (this->sysFrmCnt < this->iTotalFrames)
		{
			track_results = ReturnTrackingResults(this->sysFrmCnt, this->liveTrkVec, nTargets, 5);
			// Write tracking results as *.txt file
			WriteTrackingResults(this->cam_num, this->sysFrmCnt, nTargets, track_results, 5, FRAME_OFFSET, "", string("1"));
			this->sysFrmCnt = this->sysFrmCnt + 1;
		}
		return track_results;
	}

	if (ASSOCIATION_STAGE_2_ON) {

		// Save before Tracklet-Association
		// All Tracklets (with no repect to their length, no latency)
		int nTargetsAll = 0;
		double *track_results_stg1 = ReturnTrackingResults(this->sysFrmCnt, this->liveTrkVec, nTargetsAll, 5);
		//WriteTrackingResults(this->cam_num, this->sysFrmCnt, nTargetsAll, track_results_stg1, 5, FRAME_OFFSET, "", string("1"));
		if (STAGE_1_VISUALIZATION_ON) {
			cv::Mat img_stg1 = img.clone();
			for (int i = 0; i < this->liveTrkVec.size(); i++) {
				this->cvBoundingBox(img_stg1, this->liveTrkVec[i].rec, this->color_tab[this->liveTrkVec[i].id % 26], 2, this->liveTrkVec[i].id);
			}
			cv::imshow("STAGE 1",img_stg1);
			cv::waitKey(10);
			img_stg1.release();
		}


		cv::Mat img_latency = imgBatch[0].clone();	// image for no latent association tracking

		if (DEBUG_PRINT) cout << "(6)";
		// Put the re-arranged targets into tracklets according to ID, frame by frame
		// Insert the re-arranged tracklets to tracklets map according to ID as a key
		this->SortTrackletsbyID(this->tracksbyID, this->liveTrkVec);

		if (DEBUG_PRINT) cout << "(7)\n";

		this->ClassifyTrackletReliability(iFrmCnt, this->tracksbyID, this->tracks_reliable, this->tracks_unreliable);

		vector<BBTrk> liveReliables, lostReliables;
		this->ClassifyReliableTracklets2LiveLost(iFrmCnt, this->tracks_reliable, liveReliables, lostReliables);

		if (!ONLY_SHORT_TRACKLET_ELIMINZATION_ON) {

			if (lostReliables.size() && liveReliables.size()) {

				this->DataAssocTrkWise(iFrmCnt - FRAMES_DELAY, img_latency, lostReliables, liveReliables);
				this->ArrangeRevivedTracklets(this->tracks_reliable, liveReliables); // it can be tracks_unreliabe, liveUnreliables
			}
			if (OCC_HANDLING_TRACKLET_WISE_ON) {
				//this->CheckOcclusionsMergeStates(liveReliables, 2.0, 0.5); // do only occlusion check
				this->CheckOcclusionsGroups(liveReliables, 2.0, 0.0); // do only occlusion check
				//this->UnifyNeighborGroups(liveReliables);

				//vector<vector<RectID>> intput_groups;
				//vector<vector<RectID>> obss_groups;
				vector<BBTrk>::iterator iterR;
				if(DEBUG_PRINT)printf("\n[Occlusions]\n");
				for (iterR = liveReliables.begin(); iterR != liveReliables.end(); ++iterR) {

					if (!iterR->occTargets.empty())
					{
						//intput_groups.push_back(iterR->occTargets); // (1,2) (2,5) (1,5,2) 의 경우 하나로 묶어줄수 있어야한다. 재귀함수밖에 답이 없어보이네..

						vector<BBTrk> liveTrk = this->tracks_reliable[iterR->id];
						if (DEBUG_PRINT)printf("ID%d[%d](%d,%d,%d,%d) with ", iterR->id, (liveTrk.back().fn- liveTrk.front().fn),\
							iterR->rec.x, iterR->rec.y, iterR->rec.width, iterR->rec.height);
						cv::Rect rect_union = iterR->rec;
						vector<RectID>::iterator iterOCC;
						for (iterOCC = iterR->occTargets.begin(); iterOCC != iterR->occTargets.end(); ++iterOCC) {
							
							vector<BBTrk> liveOccTrk = this->tracks_reliable[iterOCC->nid];
							if (DEBUG_PRINT)printf("ID%d[%d](%d,%d,%d,%d), ", iterOCC->nid, (liveOccTrk.back().fn - liveOccTrk.front().fn),\
								iterOCC->rec.x, iterOCC->rec.y, iterOCC->rec.width, iterOCC->rec.height);
							rect_union = rect_union | iterOCC->rec;

							// 잘 안되면 tracks_reliable 쓰는 수밖에 근데 이거 key 접근은 속도를 저하시켜서 문제다.
							//cv::Point pt0(iterOCC->occTargetRects[0].x + iterOCC->occTargetRects[0].width / 2.0, iterOCC->occTargetRects[0].y + iterOCC->occTargetRects[0].height / 2.0);
							//cv::Point pt1(iterOCC->occTargetRects[1].x + iterOCC->occTargetRects[1].width / 2.0, iterOCC->occTargetRects[1].y + iterOCC->occTargetRects[1].height / 2.0);

						}
						if (DEBUG_PRINT)printf("\n");
						if (OCCLUSION_DETECTION_VISUALIZATION) {

							int MIN_FD = 6; // for occlusion detection
							int MIN_TL = 4; // for occlusion detection
							int TRACK_VECTOR_SIZE = liveTrk.size();
							if (/*this->tracks_reliable[iterR->id].size()*/ (liveTrk.back().fn - liveTrk.front().fn )>= MIN_FD && (TRACK_VECTOR_SIZE >=MIN_TL)) {
					
								int CONSTRAINT_LAST_IDX = TRACK_VECTOR_SIZE - 3;
								int idx_last =( CONSTRAINT_LAST_IDX > 0) ? CONSTRAINT_LAST_IDX : 0;

								cv::Rect rec_corrected = liveTrk.at(idx_last).rec;
								int fn_last = liveTrk.at(idx_last).fn;
								int xc = rec_corrected.x + rec_corrected.width / 2.0;
								int yc = rec_corrected.y + rec_corrected.height / 2.0;
					
								// instant speed		
								/*
								int fn_first = vecTrk.at(vecTrk.size() - 4).fn;
								cv::Rect rec_prev = vecTrk.at(vecTrk.size() - 4).rec;
								cv::Point pt_t_2(rec_prev.x + rec_prev.width / 2.0, rec_prev.y + rec_prev.height / 2.0);
								cv::Point pt_t_1(xc, yc);

								int vx = (pt_t_1.x - pt_t_2.x)/(fn_last-fn_first);
								int vy = (pt_t_1.y - pt_t_2.y)/(fn_last-fn_first);*/

								// average speed
								int CONSTRAINT_FIRST_IDX = TRACK_VECTOR_SIZE - MIN_FD; // liveTrk.size() 직접 호출하면 -1 일때 0이 아닌 -1이 입력되서 이렇게 바꿈
								int idx_first = (CONSTRAINT_FIRST_IDX > 0) ? CONSTRAINT_FIRST_IDX : 0;

								int fn_first = liveTrk.at(idx_first).fn;			
								cv::Rect rec_prev_average = liveTrk.at(idx_first).rec;
								cv::Point pt_t_2(rec_prev_average.x + rec_prev_average.width / 2.0, rec_prev_average.y + rec_prev_average.height / 2.0);
								cv::Point pt_t_1(xc, yc);

								int vx = (pt_t_1.x - pt_t_2.x) / (fn_last - fn_first);
								int vy = (pt_t_1.y - pt_t_2.y) / (fn_last - fn_first);
				
								int passed_frame = (liveTrk.at(liveTrk.size() - 1).fn - liveTrk.at(liveTrk.size() - 3).fn)+1; // +1 이 결정적
								this->cvBoundingBox(img_latency, rec_corrected, this->color_tab[iterR->id % 26], 1, iterR->id);
								cv::arrowedLine(img_latency, cv::Point(xc, yc), cv::Point(xc + passed_frame * vx, yc + passed_frame * vy), this->color_tab[iterR->id % 26], 1);
	
								// rec_corrected is considered as an observations in a group
								// prediction 까지 끝난 observation
								rec_corrected.x += (passed_frame * vx);
								rec_corrected.y += (passed_frame * vy);
				
								iterR->rec_corr = rec_corrected;
								this->tracks_reliable[iterR->id].back().rec_corr = iterR->rec_corr;

								this->cvBoundingBox(img_latency, rec_corrected, this->color_tab[iterR->id % 26], 2, iterR->id);	//
								this->cvBoundingBox(img_latency, rect_union, cv::Scalar(0, 0, 255), 4);							// union of the occluded targets regions
							}
						}
					}
				}
				if (DEBUG_PRINT)printf("..done\n");
				if (GROUP_UNIFYING_MANAGEMENT_ON) {
					//vector<vector<RectID>> refined_groups; // 필요하다면 class 내 global 변수로 해서 관리해야한다.
					this->UnifyNeighborGroups(liveReliables); // 구현중..

					map<int, vector<RectID>>::iterator iterG;
					for (iterG = this->groupsBatch[2].begin(); iterG != this->groupsBatch[2].end(); ++iterG) {
						if (!iterG->second.empty()) {
							int group_min_id = iterG->first;
							if (DEBUG_PRINT)printf("\n[Groups, min ID:%d]\n",group_min_id);
							cv::Rect group_rect = iterG->second.front().rec;
							vector<RectID>::iterator iterR;
							for (iterR = iterG->second.begin(); iterR != iterG->second.end(); ++iterR) {
								group_rect = group_rect | iterR->rec;
								
								if (DEBUG_PRINT)printf("ID%d(%d,%d,%d,%d)\n", iterR->nid, iterR->rec.x, iterR->rec.y, iterR->rec.width, iterR->rec.height);
							}
							this->cvBoundingBox(img_latency, group_rect, cv::Scalar(255, 255, 255), 3, group_min_id);

							// 일단 두개끼리일때만 해보자
							if (iterG->second.size() == 2) {
								RectID objs_origin[2],objs_corr[2];
								if (iterG->second.at(0).nid < iterG->second.at(1).nid) {
									objs_origin[0] = iterG->second.at(0);
									objs_origin[1] = iterG->second.at(1);
								}
								else {
									objs_origin[0] = iterG->second.at(1);
									objs_origin[1] = iterG->second.at(0);
								}
								objs_corr[0].nid = objs_origin[0].nid;
								objs_corr[0].rec = this->tracks_reliable[objs_corr[0].nid].back().rec_corr;
								objs_corr[1].nid = objs_origin[1].nid;
								objs_corr[1].rec = this->tracks_reliable[objs_corr[1].nid].back().rec_corr;

								if (DEBUG_PRINT)printf("ID%d(%d,%d,%d,%d)(corrected)\n",\
									objs_corr[0].nid, objs_corr[0].rec.x, objs_corr[0].rec.y, objs_corr[0].rec.width, objs_corr[0].rec.height);
								if (DEBUG_PRINT)printf("ID%d(%d,%d,%d,%d)(corrected)\n", \
									objs_corr[1].nid, objs_corr[1].rec.x, objs_corr[1].rec.y, objs_corr[1].rec.width, objs_corr[1].rec.height);

								// opencv: http://answers.opencv.org/question/7198/dot-product/
								// cpp: https://en.cppreference.com/w/cpp/algorithm/inner_product
								// 좀더 지능적인 모델링이 필요하겠군, 일단 AVSS 테스트해보자, 데이터 셋이 어떤지는 봐야하니까
								// ReDetPast 도 통합하고
								double cosine_similarity = 0;
								if (objs_corr[0].rec.width > 0 && objs_corr[0].rec.height > 0 && \
									objs_corr[1].rec.width > 0 && objs_corr[1].rec.height > 0) {

									std::vector<double> A{ objs_corr[0].rec.x - objs_corr[0].rec.width / 2.0 - objs_corr[1].rec.x + objs_corr[1].rec.width / 2.0,\
										objs_corr[0].rec.y - objs_corr[0].rec.height / 2.0 - objs_corr[1].rec.y + objs_corr[1].rec.height / 2.0 };
									std::vector<double> B{ objs_origin[0].rec.x - objs_origin[0].rec.width / 2.0 - objs_origin[1].rec.x + objs_origin[1].rec.width / 2.0,\
										objs_origin[0].rec.y - objs_origin[0].rec.height / 2.0 - objs_origin[1].rec.y + objs_origin[1].rec.height / 2.0 };

									double AB = A[0] * B[0] + A[1] * B[1];
									double AA = A[0] * A[0] + A[1] * A[1];
									double BB = B[0] * B[0] + B[1] * B[1];
									cosine_similarity = AB / sqrt(AA*BB);
								}
							
								if (OCCLUSION_DETECTION_VISUALIZATION) {
									if (DEBUG_PRINT)printf("%.8lf\n", cosine_similarity);
									std::ostringstream ost;
									ost << cosine_similarity;
									std::string str = ost.str();
									char cArrCosSimilarity[8];
									int c;
									for (c = 0; c < 5 && c < str.size(); c++) cArrCosSimilarity[c] = str.c_str()[c];
									cArrCosSimilarity[c] = '\0';
									cv::Point pt_cos_similarity;
									pt_cos_similarity.x = group_rect.x + 15;

									pt_cos_similarity.y = group_rect.y + group_rect.height - 15;
									cv::Scalar color;
									if (cosine_similarity <= 0) color = cv::Scalar(0, 0, 255);
									if (cosine_similarity > 0) color = cv::Scalar(255, 0, 0);

									cv::putText(img_latency, cArrCosSimilarity, pt_cos_similarity, FONT_HERSHEY_SIMPLEX, 0.4, color, 2);
									
								}
								if (cosine_similarity <0 ) {
									
									this->cvBoundingBox(img_latency, objs_origin[0].rec, this->color_tab[objs_origin[1].nid % 26], 5, objs_origin[1].nid);
									this->cvBoundingBox(img_latency, objs_origin[1].rec, this->color_tab[objs_origin[0].nid % 26], 5, objs_origin[0].nid);
									cv::waitKey(10);
									// 길이보다는 연속적으로 존재하느냐를 검사해야할듯

									cv::Rect rects[2][TRACK_MIN_LENGHT];

									// index 0: sysFrmCnt - 2
									// index 1: sysFrmCnt - 1
									// index 2: sysFrmCnt - 0
									if (DEBUG_PRINT)printf("[1]");
									for (int f = 0; f < TRACK_MIN_LENGHT; ++f) {
										rects[0][f] = cv::Rect(-1, -1, -1, -1);
										if(tracksbyID[objs_origin[0].nid].at(tracksbyID[objs_origin[0].nid].size() - f-1).fn>= sysFrmCnt - FRAMES_DELAY)
											rects[0][FRAMES_DELAY-f] = tracksbyID[objs_origin[0].nid].at(tracksbyID[objs_origin[0].nid].size() - f-1).rec;
										
										rects[1][f] = cv::Rect(-1, -1, -1, -1);
										if (tracksbyID[objs_origin[1].nid].at(tracksbyID[objs_origin[1].nid].size() - f-1).fn >= sysFrmCnt - FRAMES_DELAY)
											rects[1][FRAMES_DELAY-f] = tracksbyID[objs_origin[0].nid].at(tracksbyID[objs_origin[0].nid].size() - f-1).rec;
									}
									if (DEBUG_PRINT)printf("[2]");
									// "from sysFrmCnt - FRAMES_DELAY to sysFrmCnt" 이걸 전부 바꿔야한다. (for time t-2, t-1, t)
									// this->tracksbyID
									vector<BBTrk>::reverse_iterator rIterT;
									for (rIterT = this->tracksbyID[objs_origin[0].nid].rbegin(); rIterT != this->tracksbyID[objs_origin[0].nid].rend(); ++rIterT)
									{
										if (rIterT->fn < sysFrmCnt - FRAMES_DELAY)
											break;
										if (rIterT->fn >= sysFrmCnt - FRAMES_DELAY) {
											rIterT->isAlive = true;
											rIterT->rec = rects[1][FRAMES_DELAY - sysFrmCnt + rIterT->fn];
										}
									}
									for (rIterT = this->tracksbyID[objs_origin[1].nid].rbegin(); rIterT != this->tracksbyID[objs_origin[1].nid].rend(); ++rIterT)
									{
										if (rIterT->fn < sysFrmCnt - FRAMES_DELAY)
											break;
										if (rIterT->fn >= sysFrmCnt - FRAMES_DELAY) {
											rIterT->isAlive = true;
											rIterT->rec = rects[0][FRAMES_DELAY - sysFrmCnt + rIterT->fn];
										}
									}

									if (DEBUG_PRINT)printf("[3]");
									// (sysFrmCnt - FRAMES_DELAY)(for time t-2)
									// this->tracks_reliable
									this->tracks_reliable[objs_origin[0].nid].back().rec = objs_origin[1].rec;
									this->tracks_reliable[objs_origin[1].nid].back().rec = objs_origin[0].rec;

									if (DEBUG_PRINT)printf("[4]");
									//// 바꾼뒤 과거부터 다시 association 해야하나? ㅇㅇ 그게 제일 좋다.
									// 알았다. tracklet-association 으로 살아나서 같은 ID가 두개가 된다.
									// 그렇게 안되도록 뜯어고쳐야한다. 실마리를 찾았으니 이제 집에 가자 180823
									vector<BBTrk>::iterator iterTfw;
									bool areAllSwapped[2] = { false,false };
									for (iterTfw = liveReliables.begin(); iterTfw != liveReliables.end(); ++iterTfw) {
										if (iterTfw->id == objs_origin[0].nid) {
											iterTfw->rec = objs_origin[1].rec;
											areAllSwapped[0] = true;
										}
										if (iterTfw->id == objs_origin[1].nid) {
											iterTfw->rec = objs_origin[0].rec;
											areAllSwapped[1] = true;
										}
										if (areAllSwapped[0] && areAllSwapped[1])
											break;
									}
									if (DEBUG_PRINT)printf("[5]");
									// this->liveTrkVec (for time t)
									// frame-wise (no latency),
									areAllSwapped[0] = false;
									areAllSwapped[1] = false;
									// 이 둘중 분명 isAlive = false 여서 lostTrkVec 으로 넘어간놈이 있을것이다. 즉, areAllSwapped[0] && areAllSwapped[1] 이 경우는 나오지 않는다.
									for (iterTfw = this->liveTrkVec.begin(); iterTfw != this->liveTrkVec.end(); ++iterTfw) {
										if (iterTfw->id == objs_origin[0].nid) {
											iterTfw->rec = rects[1][FRAMES_DELAY];
											iterTfw->isAlive = true;
											areAllSwapped[0] = true;
										}
										if (iterTfw->id == objs_origin[1].nid) {
											iterTfw->rec = rects[0][FRAMES_DELAY];
											iterTfw->isAlive = true;
											areAllSwapped[1] = true;
										}
										if (areAllSwapped[0] && areAllSwapped[1])
											break;
									}
									if (!areAllSwapped[0]) { // not found
										BBTrk bbt;
										bbt.isAlive = true;
										bbt.id = objs_origin[0].nid;
										bbt.fn = iFrmCnt;
										bbt.rec = rects[1][FRAMES_DELAY];
										bbt.vx = this->tracksbyID[objs_origin[0].nid].back().vx;
										bbt.vy = this->tracksbyID[objs_origin[0].nid].back().vy;
										bbt.weight = this->tracksbyID[objs_origin[0].nid].back().weight;
										this->liveTrkVec.push_back(bbt);
									}
									if (!areAllSwapped[1]) { // not found
										BBTrk bbt;
										bbt.isAlive = true;
										bbt.id = objs_origin[1].nid;
										bbt.fn = iFrmCnt;
										bbt.rec = rects[0][FRAMES_DELAY];
										bbt.vx = this->tracksbyID[objs_origin[0].nid].back().vx;
										bbt.vy = this->tracksbyID[objs_origin[0].nid].back().vy;
										bbt.weight = this->tracksbyID[objs_origin[0].nid].back().weight;
										this->liveTrkVec.push_back(bbt);
									}
									cv::Mat img_stg1 = img.clone();
									for (int i = 0; i < this->liveTrkVec.size(); i++) {
										this->cvBoundingBox(img_stg1, this->liveTrkVec[i].rec, this->color_tab[this->liveTrkVec[i].id % 26], 2, this->liveTrkVec[i].id);
									}
									//cv::imshow("STAGE 1", img_stg1);
									cv::waitKey(10);
									img_stg1.release();
									if (DEBUG_PRINT)printf("[6]");
									// this->liveTracksBatch (for time t-2, t-1, t)
									for (int b = 0;b<TRACK_MIN_LENGHT;++b) {
										areAllSwapped[0] = false;
										areAllSwapped[1] = false;
										for (iterTfw = this->liveTracksBatch[b].begin(); iterTfw != this->liveTracksBatch[b].end(); ++iterTfw) {
											if (iterTfw->id == objs_origin[0].nid) {
												iterTfw->rec = rects[1][b];
												areAllSwapped[0] = true;
											}
											if (iterTfw->id == objs_origin[1].nid) {
												iterTfw->rec = rects[0][b];
												areAllSwapped[1] = true;
											}
											if (areAllSwapped[0] && areAllSwapped[1])
												break;
										}
									}
								}

							}

							// 세개이상일땐 관계를 어떻게 모델링해야하나.. 결국 combitorial일수밖에 없나? 2depth 식으로 polynomial 비용으로 대강 해결하자
						}
					}
				}

				// Group 내애서 
				// Group 별 multi-processing이 가능하다
				// frame-wise data association, 속도 고려할지 여부는 결과 보고 결정, 이건 사실 full occlusion일 때만 수행하면 될듯(나중에 구현한 이후에)

				// 여러 방안이 있다.
				// (1) occlusion 일때 local search 로 ID switch 를 방지  
				// -> 잘된다면, 멈춰있는 객체에 대해서도 잘 작동	
				// (2) 아예 frame-wise association 이 수행되지 않도록 함. 
				// -> 멈취있는 객체는 못 발견 할듯, motion을 처음-끝 의 평균으로 해서, 
				// -> 여기에 가속도를 반영한 속도 추정이 필요할듯, 된다면 오다가 멈추는 객체도 판별가능
				// -> 아니면, 급격한 움직임이 중간에 있으면 거기부터 새로 모션추정
				// 상호 보완으로 적용해야 하는가?
			}

			if(MOTION_PREDICTION_VISUALIZATION){

				// QuadraticMotionEstimation(lostReliables);
				vector<vector<Point>> stats_pts; // for parallel processing
				//cv::Mat img_latency = this->imgBatch[0].clone();
				// Access Map by stats' ID
				vector<BBTrk>::const_iterator iterLost;
				for (iterLost = lostReliables.begin(); iterLost != lostReliables.end(); iterLost++)
				{
					if (tracks_reliable[iterLost->id].back().fn > this->sysFrmCnt - FRAMES_DELAY - 30) {

						float fl = tracks_reliable[iterLost->id].back().fn - tracks_reliable[iterLost->id].front().fn;
						cv::Rect r1, r2;
						cv::Point2f p1, p2;

						r1 = tracks_reliable[iterLost->id].front().rec;
						r2 = tracks_reliable[iterLost->id].back().rec;

						p1 = cv::Point2f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0);
						p2 = cv::Point2f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0);

						float vx = (p2.x - p1.x) / fl;
						float vy = (p2.y - p1.y) / fl;

						vector<Point> pts;
						vector<BBTrk>::iterator iterT;
						for (iterT = tracks_reliable[iterLost->id].begin(); iterT != tracks_reliable[iterLost->id].end(); iterT++) {
							pts.push_back(cv::Point(iterT->rec.x + iterT->rec.width / 2.0, iterT->rec.x + iterT->rec.height / 2.0));
						}
						// Draw the Bounding Box and the Past Tracklet 
						this->cvBoundingBox(img_latency, r2, this->color_tab[iterLost->id % 26], 1, iterLost->id);
						//cv::polylines(img_latency, pts, false, this->color_tab[iterLost->id % 26]);

						vector<BBTrk>::const_iterator iterLive;
						for (iterLive = liveReliables.begin(); iterLive != liveReliables.end(); iterLive++) {

							this->cvBoundingBox(img_latency, iterLive->rec, this->color_tab[iterLive->id % 26], 2, iterLive->id);

							float fd = tracks_reliable[iterLive->id].front().fn - tracks_reliable[iterLost->id].back().fn;

							if (fd > 0) {
								// Draw the Predicted Location and Linearly Estimated mMtion
								cv::arrowedLine(img_latency, p2, cv::Point2f(p2.x + vx*fd, p2.y + vy*fd), this->color_tab[iterLost->id % 26], 2);
								//cv::line(img_latency, p2, cv::Point2f(p2.x + vx*fd, p2.y + vy*fd), this->color_tab[iterLost->id % 26]);
							}
						}
					}
				}
			}
			if (OCCLUSION_DETECTION_VISUALIZATION||MOTION_PREDICTION_VISUALIZATION) {

				//SaveAImage((this->sysFrmCnt - FRAMES_DELAY), &IplImage(img_latency), "res\\trk_disp\\motion_", "jpg");
				//cv::imshow("Tracklet Association", img_latency);
				//cv::waitKey(1);
				img_latency.release();
			}
		}
		detVec.clear();

		double* track_results_all = nullptr;
		double* track_results_reliable = nullptr;
		if ( this->sysFrmCnt < this->iTotalFrames)
		{
			if(this->sysFrmCnt - FRAMES_DELAY >= 0){
			// All Tracklets (with no repect to their length, no latency)
			int nTargetsAll = 0;

			// Reliable Tracklets (with the minimum length over FRAMES_DELAY, with latency equal to FRAMES_DELAY)
			track_results_reliable = ReturnTrackingResults(this->sysFrmCnt - FRAMES_DELAY, liveReliables, nTargets, 5);
			this->allLiveReliables.push_back(liveReliables);
			//WriteTrackingResults(this->cam_num, this->sysFrmCnt - FRAMES_DELAY, nTargets, track_results_reliable, 5, FRAME_OFFSET, "", string("1_latency"),Th_Conf);
			}
			this->sysFrmCnt++;
		}

		if (this->sysFrmCnt == this->iTotalFrames) {
			this->sysFrmCnt--; // 반드시 해줘야함

			if (DEBUG_PRINT) printf("Post-Processing for the latent tracking results..");
			for (int OFFSET = 1; OFFSET < TRACK_MIN_LENGHT; OFFSET++) {
				double* latent_targets_reliable;
				latent_targets_reliable = ReturnTrackingResults(this->sysFrmCnt, this->liveTracksBatch[OFFSET], liveReliables, nTargets, 5);
				this->allLiveReliables.push_back(liveReliables);
				// Write tracking results as *.txt file
				//WriteTrackingResults(this->cam_num, this->sysFrmCnt - FRAMES_DELAY + OFFSET, nTargets, latent_targets_reliable, 5, FRAME_OFFSET, "", string("1_latency"), Th_Conf);
			}
			if (DEBUG_PRINT) printf("done.\n");
		}

		if ((this->sysFrmCnt - FRAMES_DELAY > TRACK_CLEARING_TERM) && ((this->sysFrmCnt - FRAMES_DELAY)%TRACK_ASSOCIATION_TERM==0)) {
			ClearOldEmptyTracklet(this->sysFrmCnt - FRAMES_DELAY, this->tracksbyID);
			ClearOldEmptyTracklet(this->sysFrmCnt - FRAMES_DELAY, this->tracks_reliable);
			ClearOldEmptyTracklet(this->sysFrmCnt - FRAMES_DELAY, this->tracks_unreliable);
		}

		return track_results_reliable;


	}
}
void SYM_MOT_HGMPHD::PushTargetsVecs2BatchesLiveLost() {
	if (this->sysFrmCnt >= TRACK_MIN_LENGHT) {
		for (int q = 0; q < FRAMES_DELAY; q++) {
			for (int i = 0; i < this->liveTracksBatch[q].size(); i++)this->liveTracksBatch[q].at(i).Destroy();
			this->liveTracksBatch[q].clear();
			this->liveTracksBatch[q] = liveTracksBatch[q + 1];

			for (int i = 0; i < this->lostTracksBatch[q].size(); i++)this->lostTracksBatch[q].at(i).Destroy();
			this->lostTracksBatch[q].clear();
			this->lostTracksBatch[q] = lostTracksBatch[q + 1];
		}
		this->liveTracksBatch[FRAMES_DELAY] = this->liveTrkVec;
		this->lostTracksBatch[FRAMES_DELAY] = this->lostTrkVec;
	}
	else if (this->sysFrmCnt < TRACK_MIN_LENGHT) {
		this->liveTracksBatch[this->sysFrmCnt] = this->liveTrkVec;
		this->lostTracksBatch[this->sysFrmCnt] = this->lostTrkVec;
	}
}

void SYM_MOT_HGMPHD::ArrangeTargetsVecsBatchesLiveLost() {
	vector<BBTrk> liveTargets;
	vector<BBTrk> lostTargets;
	for (int tr = 0; tr < this->liveTrkVec.size(); ++tr) {
		if (this->liveTrkVec[tr].isAlive) {
			liveTargets.push_back(this->liveTrkVec[tr]);
			// 이럼 new birth도 포함되네, 이건 merge에서 쓸데없는 순환을 발생시킬텐데.. 이럼안된다. 
			// 결국 DataAssocLow 안에 contextMap 기반 association 을 넣어줘야한다.
		}
		else if (!this->liveTrkVec[tr].isAlive /*&& !this->trkVec[tr].isMerged*/) {
			//if (iFrmCnt == 37 && this->trkVec[tr].id == 27) {
			//	printf("ID27(%d, %d)\n", this->trkVec[tr].isAlive, this->trkVec[tr].isMerged);
			//	cv::waitKey();
			//}
			lostTargets.push_back(this->liveTrkVec[tr]);
		}
		else {
			// abandon the merged targets (When target a'ID and b'TD are merged with a'ID < b'TD, target b is abandoned and not considered as LB_ASSOCIATION) 
		}
	}
	this->liveTrkVec.swap(liveTargets);	// swapping the alive targets
	this->lostTrkVec.swap(lostTargets);	// swapping the loss tragets	
	liveTargets.clear();
	lostTargets.clear();
}
void SYM_MOT_HGMPHD::SortTrackletsbyID(map<int, vector<BBTrk>>& tracksbyID, vector<BBTrk>& targets) {
	pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty;
	for (int j = 0; j < targets.size(); j++)
	{
		int id = targets.at(j).id;

		// targets.at(j).fn = this->sysFrmCnt; // 이게 왜 안넘어 갔는지 미스테리다, 와.. prediction 에서 framenumber를 update 안해줬네..

		vector<BBTrk> tracklet;
		tracklet.push_back(targets.at(j));

		pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = tracksbyID.insert(map<int, vector<BBTrk>>::value_type(id, tracklet));

		if (isEmpty.second == false) { // already has a element with target.at(j).id
			tracksbyID[id].push_back(targets.at(j));
			//if (DEBUG_PRINT)
				//printf("[%d-%d]ID%d is updated into tracksbyID\n", this->sysFrmCnt, targets.at(j).fn, id);
		}
		else {
			//if (DEBUG_PRINT)
				//printf("[%d-%d]ID%d is newly added into tracksbyID\n",this->sysFrmCnt, targets.at(j).fn, id);
		}

	}
}
void SYM_MOT_HGMPHD::ClassifyTrackletReliability(int iFrmCnt, map<int, vector<BBTrk>>& tracksbyID, map<int, vector<BBTrk>>& reliables, map<int, std::vector<BBTrk>>& unreliables) {

	map<int, vector<BBTrk>>::iterator iterID;
	//if (DEBUG_PRINT) printf("[%d]", iFrmCnt);

	for (iterID = tracksbyID.begin(); iterID != tracksbyID.end(); iterID++) {
		if (!iterID->second.empty()) {
			//if (DEBUG_PRINT) printf("ID%d(%d-%d, size:%d) ", iterID->first, iterID->second.front().fn, iterID->second.back().fn, iterID->second.size());
			if (iterID->second.back().fn == iFrmCnt) {										// live 인것만 누적시켜도 완성된다. online 이니까

				vector<BBTrk> tracklet;
				vector<BBTrk>::reverse_iterator rIterT;
				bool isFound = false;
				for (rIterT = iterID->second.rbegin(); rIterT != iterID->second.rend(); rIterT++) {
					if (rIterT->fn == iFrmCnt - FRAMES_DELAY) { // 1,2,3,4,7,8,9,10,11 같은 경우는 일단 고려하지 않음

						tracklet.push_back(rIterT[0]);
						isFound = true;
						break;
					}
				}

				if (isFound /*iterID->second.back().fn - iterID->second.front().fn >= FRAMES_DELAY*/) { // reliable (with latency)
					pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = reliables.insert(map<int, vector<BBTrk>>::value_type(iterID->first, tracklet));
					if (isEmpty.second == false) {
						reliables[iterID->first].push_back(tracklet[0]);
						//printf("[%d(%d)]ID%d(%d) is reliable\n", iFrmCnt - FRAMES_DELAY, tracklet[0].fn, iterID->first,tracklet[0].id);
					}

					unreliables[iterID->first].clear();

					//if (DEBUG_PRINT)
						//printf("[%d]ID%d is added into reliables\n", this->sysFrmCnt, iterID->first);
				}
				else {																					// unreliable (witout latency)
					pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = unreliables.insert(map<int, vector<BBTrk>>::value_type(iterID->first, iterID->second));
					if (isEmpty.second == false)
						unreliables[iterID->first].push_back(iterID->second.back());

					//if (DEBUG_PRINT)
						//printf("[%d]ID%d is added into unreliables\n", this->sysFrmCnt, iterID->first);
				}
			}

		}

	}


	/*vector<BBTrk>::iterator iterT;
	for (iterT = this->liveTrkVec.begin(); iterT != this->liveTrkVec.end(); iterT++) {

		if(!tracksbyID[iterT->id].empty() && !tracksbyID[iterT->id].

	}*/

	/*else {
		printf("ClassifyTrackletReliability has been wrongly called!!\n");
		exit(-1);
	}*/
}
void SYM_MOT_HGMPHD::ClassifyReliableTracklets2LiveLost(int iFrmCnt, const map<int, vector<BBTrk>>& reliables, vector<BBTrk>& liveReliables, vector<BBTrk>& LostReliables) {

	map<int, vector<BBTrk>>::const_iterator iterT;
	for (iterT = reliables.begin(); iterT != reliables.end(); iterT++) {
		if (!iterT->second.empty()) {
			if (iterT->second.back().fn == iFrmCnt - FRAMES_DELAY) {
				if (DEBUG_PRINT)
					printf("[%d]ID%d(%d,%d,%d,%d) is reliable live\n", iFrmCnt - FRAMES_DELAY, iterT->first, iterT->second.back().rec.x, iterT->second.back().rec.y, iterT->second.back().rec.width, iterT->second.back().rec.height);
				liveReliables.push_back(iterT->second.back());
			}
			else if (iterT->second.back().fn < iFrmCnt - FRAMES_DELAY) {
				LostReliables.push_back(iterT->second.back());
			}
		}
	}
}
void SYM_MOT_HGMPHD::ArrangeRevivedTracklets(map<int, vector<BBTrk>>& tracks, vector<BBTrk>& lives) {

	// ID Management
	vector<BBTrk>::iterator iterT;
	for (iterT = lives.begin(); iterT != lives.end(); ++iterT) {
		if (iterT->id_associated >= 0) { // id != -1, succeed in ID recovery;

			// input parameter 1: tracks
			int size_old = tracks[iterT->id_associated].size();
			tracks[iterT->id_associated].insert(tracks[iterT->id_associated].end(), tracks[iterT->id].begin(), tracks[iterT->id].end());
			int size_new = tracks[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i)
				tracks[iterT->id_associated].at(i).id = iterT->id_associated;
			tracks[iterT->id].clear();

			// this->tracksbyID
			size_old = this->tracksbyID[iterT->id_associated].size();
			this->tracksbyID[iterT->id_associated].insert(this->tracksbyID[iterT->id_associated].end(), this->tracksbyID[iterT->id].begin(), this->tracksbyID[iterT->id].end());
			size_new = this->tracksbyID[iterT->id_associated].size();
			for (int i = size_old; i < size_new; ++i)
				this->tracksbyID[iterT->id_associated].at(i).id = iterT->id_associated;
			this->tracksbyID[iterT->id].clear();

			// this->liveTrkVec (no letancy tracking)
			vector<BBTrk>::iterator iterTfw; // frame-wise (no latency)
			for (iterTfw = this->liveTrkVec.begin(); iterTfw != this->liveTrkVec.end(); ++iterTfw) {
				if (iterTfw->id == iterT->id) {
					iterTfw->id = iterT->id_associated;
					break;
				}
			}
			// this->liveTracksBatch (at t-2, t-1, t)
			for (int b = 0; b < TRACK_MIN_LENGHT; ++b) {
				for (iterTfw = this->liveTracksBatch[b].begin(); iterTfw != this->liveTracksBatch[b].end(); ++iterTfw) {
					if (iterTfw->id == iterT->id) {
						iterTfw->id = iterT->id_associated;
						break;
					}
				}
			}

			// input parameter 2: lives
			iterT->id = iterT->id_associated;
		}
	}
}
void SYM_MOT_HGMPHD::CheckOcclusionsMergeStates(vector<BBTrk>& stats, const double T_merge, const double T_occ) {

	double MERGE_THRESHOLD = T_merge;
	double OCCLUSION_THRESHOLD = T_occ;
	if (!MERGE_ON)						MERGE_THRESHOLD = 2.0;		// do not merge any target
	//if (!OCC_HANDLING_FRAME_WISE_ON)	OCCLUSION_THRESHOLD = 2.0;  // do not handle any occlusion

	vector<vector<int>> mergeStatsIdxes(stats.size());
	mergeStatsIdxes.resize(stats.size());
	vector<vector<double>> mergeStatsORs(stats.size()); // OR: Overlapping Ratio [0.0, 2.0]
	mergeStatsORs.resize(stats.size());

	vector<vector<bool>> visitTable;
	visitTable.resize(stats.size(), std::vector<bool>(stats.size(), false));
	for (int v = 0; v < stats.size(); v++) visitTable[v][v] = true;

	int* mergeIdxTable = new int[stats.size()];
	for (int i = 0; i < stats.size(); i++) mergeIdxTable[i] = -1;

	double* overlapRatioTable = new double[stats.size()];
	for (int i = 0; i < stats.size(); i++) overlapRatioTable[i] = 0.0;

	for (int a = 0; a < stats.size(); ++a) {

		// clearing before checking occlusion
		stats.at(a).occTargets.clear();

		if (stats.at(a).isAlive) {

			cv::Rect Ra = stats.at(a).rec;
			cv::Point Pa = cv::Point(Ra.x + Ra.width / 2, Ra.y + Ra.height / 2);
			//cv::rectangle(distImg, Ra, cv::Scalar(255, 255, 255), 2);

			for (int b = a + 1; b < stats.size(); ++b) {
				if (stats.at(b).isAlive && !visitTable[a][b] && !visitTable[b][a]) { // if a pair is not yet visited


					cv::Rect Rb = stats.at(b).rec;
					cv::Point Pb = cv::Point(Rb.x + Rb.width / 2, Rb.y + Rb.height / 2);
					//cv::line(distImg, Pa, Pb, cv::Scalar(0, 0, 255));
					//cv::rectangle(distImg, Rb, cv::Scalar(255, 0, 0), 2);

					// check overlapping region
					double Ua = (double)(Ra & Rb).area() / (double)Ra.area();
					double Ub = (double)(Ra & Rb).area() / (double)Rb.area();
					double overlap_r = Ua + Ub; // Symmetric, The Sum-of-Intersection-over-Area (SIOA)

					double IOU = (double)(Ra & Rb).area() / (double)(Ra | Rb).area();

					// Size condition					
					if((Ra.area()>(Rb.area()*2))|| (Rb.area() > (Ra.area() * 2))) overlap_r = 0.0;

					//char carrDist[10]; sprintf(carrDist,"%.lf",dist);
					//cv::putText(distImg, string(carrDist), cv::Point((Pa.x + Pb.x) / 2, (Pa.y + Pb.y) / 2), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
					if (overlap_r >= MERGE_THRESHOLD/*IOU>=0.5*/) {


						mergeStatsIdxes[a].push_back(b);
						mergeStatsIdxes[b].push_back(a);

						mergeStatsORs[a].push_back(overlap_r);
						mergeStatsORs[b].push_back(overlap_r);

						// Store minimum distance
						//overlapRatioTable[b] = overlap_r;
						//overlapRatioTable[a] = overlap_r;
					}
					else if (overlap_r < MERGE_THRESHOLD && overlap_r>OCCLUSION_THRESHOLD/* IOU < 0.5 && IOU > 0.0*/) { // occlusion
						stats.at(a).isOcc = true;
						stats.at(b).isOcc = true;
						stats.at(a).occTargets.push_back(RectID(stats.at(b).id, stats.at(b).rec));
						stats.at(b).occTargets.push_back(RectID(stats.at(a).id, stats.at(a).rec));
					}

					// check visiting
					visitTable[a][b] = true;
					visitTable[b][a] = true;
				}
			}
			//cv::waitKey();

			// final check
			if (!stats.at(a).occTargets.empty()) {
				stats.at(a).isOcc = true;
			}
			else {
				stats.at(a).isOcc = false;
			}
		}
	}
	//cv::imshow("Distance", distImg);

	if (DEBUG_PRINT_MERGE) printf("[%d]Merge Table: ", this->sysFrmCnt);
	for (int a = 0; a < stats.size(); ++a) {
		if (stats.at(a).isAlive && !mergeStatsIdxes[a].empty()) {

			int min_id = stats.at(a).id;
			for (int m = 0; m < mergeStatsIdxes[a].size(); m++) {
				if (min_id > stats[mergeStatsIdxes[a][m]].id)
					min_id = stats[mergeStatsIdxes[a][m]].id;

				if (DEBUG_PRINT_MERGE && (mergeStatsORs[a][m] > 0.0))
					printf("%4d(%d)[%.3lf] ", stats.at(a).id, stats.at(mergeStatsIdxes[a][m]).id, mergeStatsORs[a][m]);
			}

			//printf("(the oldest ID:%d)", min_id);
			if (stats.at(a).id == min_id) {

				/*if (VIEW_LBA_2 && T_overlap == 0.999) {
				this->DrawTrkBBS(this->detailViewImg, stats.at(mergeIdxTable[a]).rec, stats.at(mergeIdxTable[a]).id, cvScalar(255, 255, 255), 2);
				this->DrawTrkBBS(this->detailViewImg, stats.at(a).rec, stats.at(a).id, cvScalar(200,200,200), 2);
				this->DrawTrkBBS(this->detailViewImg, stats.at(mergeIdxTable[a]).rec &stats.at(a).rec, -1, cvScalar(0, 0, 0), 2);
				}*/
				for (int m = 0; m < mergeStatsIdxes[a].size(); m++) {
					stats.at(a).rec.x = 0.1*stats.at(mergeStatsIdxes[a][m]).rec.x + 0.9*stats.at(a).rec.x;
					stats.at(a).rec.y = 0.1*stats.at(mergeStatsIdxes[a][m]).rec.y + 0.9*stats.at(a).rec.y;
					stats.at(a).rec.width = 0.1*stats.at(mergeStatsIdxes[a][m]).rec.width + 0.9*stats.at(a).rec.width;
					stats.at(a).rec.height = 0.1*stats.at(mergeStatsIdxes[a][m]).rec.height + 0.9*stats.at(a).rec.height;
					//this->DrawTrkBBS(this->detailViewImg, stats.at(mergeStatsIdxes[a][m]).rec, stats.at(mergeStatsIdxes[a][m]).id, this->color_tab[stats.at(mergeStatsIdxes[a][m]).id % 26], 2);
				}
				/*if (VIEW_LBA_2 &&T_overlap == 0.999)*/
				//this->DrawTrkBBS(this->detailViewImg, stats.at(a).rec, stats.at(a).id, this->color_tab[stats.at(a).id % 26], 3);

				stats.at(a).isMerged = false;	// Other IDs are merged into stats[a]
				stats.at(a).isAlive = true;
			}
			else {
				//this->DrawTrkBBS(this->detailViewImg, stats.at(a).rec, stats.at(a).id, this->color_tab[stats.at(a).id %26], 2);
				stats.at(a).isAlive = false;
				stats.at(a).isMerged = true;	// stats[a] is merged into an oldest state having the smallest ID
			}
			//cv::waitKey();
		}
	}
 
	if (DEBUG_PRINT_MERGE) printf("\n");
	delete[]mergeIdxTable;
	delete[]overlapRatioTable;
}
void SYM_MOT_HGMPHD::CheckOcclusionsGroups(vector<BBTrk>& stats, const double T_merge, const double T_occ) {

	double MERGE_THRESHOLD = T_merge;
	double OCCLUSION_THRESHOLD = T_occ;
	if (!MERGE_ON)						MERGE_THRESHOLD = 2.0;		// do not merge any target
	//if (!OCC_HANDLING_FRAME_WISE_ON)	OCCLUSION_THRESHOLD = 2.0;  // do not handle any occlusion

	vector<vector<int>> mergeStatsIdxes(stats.size());
	mergeStatsIdxes.resize(stats.size());
	vector<vector<double>> mergeStatsORs(stats.size()); // OR: Overlapping Ratio [0.0, 2.0]
	mergeStatsORs.resize(stats.size());

	vector<vector<bool>> visitTable;
	visitTable.resize(stats.size(), std::vector<bool>(stats.size(), false));
	for (int v = 0; v < stats.size(); v++) visitTable[v][v] = true;

	//int* mergeIdxTable = new int[stats.size()];
	//for (int i = 0; i < stats.size(); i++) mergeIdxTable[i] = -1;

	//double* overlapRatioTable = new double[stats.size()];
	//for (int i = 0; i < stats.size(); i++) overlapRatioTable[i] = 0.0;
	
	// Init & clearing before checking occlusion
	for (int a = 0; a < stats.size(); ++a){
		stats.at(a).isOcc = false;
		stats.at(a).occTargets.clear();
	}

	for (int a = 0; a < stats.size(); ++a) {

		if (stats.at(a).isAlive) {

			cv::Rect Ra = stats.at(a).rec;
			cv::Point Pa = cv::Point(Ra.x + Ra.width / 2, Ra.y + Ra.height / 2);
			//cv::rectangle(distImg, Ra, cv::Scalar(255, 255, 255), 2);

			for (int b = a + 1; b < stats.size(); ++b) {
				if (stats.at(b).isAlive && !visitTable[a][b] && !visitTable[b][a]) { // if a pair is not yet visited


					cv::Rect Rb = stats.at(b).rec;
					cv::Point Pb = cv::Point(Rb.x + Rb.width / 2, Rb.y + Rb.height / 2);
					//cv::line(distImg, Pa, Pb, cv::Scalar(0, 0, 255));
					//cv::rectangle(distImg, Rb, cv::Scalar(255, 0, 0), 2);

					// check overlapping region
					double Ua = (double)(Ra & Rb).area() / (double)Ra.area();
					double Ub = (double)(Ra & Rb).area() / (double)Rb.area();
					double overlap_r = Ua + Ub; // Symmetric

					double IOU = (double)(Ra & Rb).area() / (double)(Ra | Rb).area();

					// Size condition					
					if((Ra.area()>(Rb.area()*2))|| (Rb.area() > (Ra.area() * 2))) overlap_r = 0.0;

												//char carrDist[10]; sprintf(carrDist,"%.lf",dist);
												//cv::putText(distImg, string(carrDist), cv::Point((Pa.x + Pb.x) / 2, (Pa.y + Pb.y) / 2), CV_FONT_HERSHEY_COMPLEX, 0.5, cv::Scalar(0, 0, 255), 2);
					//if (overlap_r >= MERGE_THRESHOLD) {


					//	mergeStatsIdxes[a].push_back(b);
					//	mergeStatsIdxes[b].push_back(a);

					//	mergeStatsORs[a].push_back(overlap_r);
					//	mergeStatsORs[b].push_back(overlap_r);

					//	// Store minimum distance
					//	//overlapRatioTable[b] = overlap_r;
					//	//overlapRatioTable[a] = overlap_r;
					//}
					//else 

					if (overlap_r < MERGE_THRESHOLD && overlap_r>OCCLUSION_THRESHOLD/*IOU<0.5 && IOU>0.0*/) { // occlusion
						stats.at(a).isOcc = true;
						stats.at(b).isOcc = true;
						stats.at(a).occTargets.push_back(RectID(stats.at(b).id, stats.at(b).rec));
						stats.at(b).occTargets.push_back(RectID(stats.at(a).id, stats.at(a).rec));
					}

					// check visiting
					visitTable[a][b] = true;
					visitTable[b][a] = true;
				}
			}
			//cv::waitKey();

			// final check
			if (!stats.at(a).occTargets.empty()) {
				stats.at(a).isOcc = true;
			}
			else {
				stats.at(a).isOcc = false;
			}
		}

		// Find the minimum id within an occlusion group (key for groups' map container)
		int min_id = stats.at(a).id;
		vector<RectID>::iterator iterR;
		for (iterR = stats.at(a).occTargets.begin(); iterR != stats.at(a).occTargets.end(); ++iterR) {
			if (min_id > iterR->nid) min_id = iterR->nid;
		}
		for (iterR = stats.at(a).occTargets.begin(); iterR != stats.at(a).occTargets.end(); ++iterR) {
			iterR->min_id = min_id;
		}
	}

	//delete[]mergeIdxTable;
	//delete[]overlapRatioTable;
}
void SYM_MOT_HGMPHD::ClearOldEmptyTracklet(int current_fn, map<int, vector<BBTrk>>& tracklets,int MAXIMUM_OLD) {
	
	map<int, vector<BBTrk>> cleared_tracklets;

	vector<int> keys_old_vec;
	map<int, vector<BBTrk>>::iterator iter;
	//if(DEBUG_PRINT)
	printf("[%d]checking %d tracks..->", current_fn, tracklets.size());
	for (iter = tracklets.begin(); iter!= tracklets.end(); ++iter) {

		if (!iter->second.empty()) {
			if (iter->second.back().fn >= current_fn - MAXIMUM_OLD) {

				vector<BBTrk> track;

				vector<BBTrk>::iterator iterT;
				for (iterT = iter->second.begin(); iterT != iter->second.end(); ++iterT)
					track.push_back(iterT[0]);

				pair<map<int, vector<BBTrk>>::iterator, bool> isEmpty = cleared_tracklets.insert(map<int, vector<BBTrk>>::value_type(iter->first, track));

				if (isEmpty.second == false) { // already exists, 실행될리는 없겠지만.

				}
				else {

				}
			}
			else {
				keys_old_vec.push_back(iter->first);
			}
		}
	}
	
	// Swap and Clear Old Tracklets
	cleared_tracklets.swap(tracklets);
	for (iter = cleared_tracklets.begin(); iter != cleared_tracklets.end(); ++iter) {
		iter->second.clear();
	}
	cleared_tracklets.clear();

	//if (DEBUG_PRINT)
	printf("clear %d tracks -> %d tracks\n", keys_old_vec.size(), tracklets.size());

}
void SYM_MOT_HGMPHD::UnifyNeighborGroups(vector<BBTrk> input_targets) {

	map<int, vector<RectID>> groups;

	// Iterate the live objects
	vector<BBTrk>::iterator iterT;
	for (iterT = input_targets.begin(); iterT != input_targets.end(); ++iterT) {
		
		// Iterate the occluded objects vector of an object
		if(!iterT->occTargets.empty()){

			int key_min_id = FindMinIDofNeigbors2Depth(input_targets, iterT->occTargets, iterT->occTargets[0].min_id); // recursive function
			
			vector<RectID> occ_group; 

			pair<map<int, vector<RectID>>::iterator, bool> isEmpty = groups.insert(map<int, vector<RectID>>::value_type(key_min_id, occ_group));

			if (isEmpty.second == false) { // already exists
				
				vector<RectID>::iterator iterR;
				for (iterR = iterT->occTargets.begin(); iterR != iterT->occTargets.end(); ++iterR)
				{
					bool isDuplicated = false;
					vector<RectID>::iterator iterG;
					for (iterG = groups[key_min_id].begin(); iterG != groups[key_min_id].end(); ++iterG) {
						if (iterG->nid == iterR->nid) {
							isDuplicated = true;
							break;
						}
					}
					if (!isDuplicated) {
						occ_group.push_back(iterR[0]);
					}
				}
				groups[key_min_id].insert(groups[key_min_id].end(), occ_group.begin(), occ_group.end());
				
				/*vector<RectID>::iterator iterRtemp;
				for (iterRtemp = occ_group.begin(); iterRtemp != occ_group.end(); ++iterRtemp)
				{
					printf("ID%d(%d,%d,%d,%d)(1)\n", iterRtemp->nid, iterRtemp->occTargetRects[0].x, iterRtemp->occTargetRects[0].y, iterRtemp->occTargetRects[0].width, iterRtemp->occTargetRects[0].height);
				}*/
			}  
			else {							// newly added
				groups[key_min_id] = iterT->occTargets;
				vector<RectID>::iterator iterRtemp;
				/*for (iterRtemp = iterT->occTargets.begin(); iterRtemp != iterT->occTargets.end(); ++iterRtemp)
				{
					printf("ID%d(%d,%d,%d,%d)(2)\n", iterRtemp->nid, iterRtemp->occTargetRects[0].x, iterRtemp->occTargetRects[0].y, iterRtemp->occTargetRects[0].width, iterRtemp->occTargetRects[0].height);
				}*/
			}
		}
		std::array<size_t, 2> shape;
	}


	// Push the queue of the groups
	this->groupsBatch[0].clear();
	this->groupsBatch[0] = this->groupsBatch[1];
	this->groupsBatch[1].clear();
	this->groupsBatch[1] = this->groupsBatch[2];
	this->groupsBatch[2].clear();
	this->groupsBatch[2] = groups;

}
int SYM_MOT_HGMPHD::FindMinIDofNeigborsRecursive(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id) {

	int new_min_id = parent_occ_group_min_id;
	vector<RectID>::iterator iterR;
	for (iterR = occ_targets.begin();iterR!=occ_targets.end(); ++iterR) {
		
		vector<BBTrk>::iterator iterT;
		for (iterT = targets.begin();iterT!= targets.end();++iterT) {
			
			
			if (iterR->nid == iterT->id) {
				if(new_min_id > iterT->occTargets[0].min_id) { // when mininmum IDs in occlusion group are different
					vector<RectID> occ_targets_not_shared = iterT->occTargets;
					new_min_id = FindMinIDofNeigborsRecursive(targets, occ_targets_not_shared, iterT->occTargets[0].min_id);
				}
				break;
			}


		}
	}
	return new_min_id;
}
int SYM_MOT_HGMPHD::FindMinIDofNeigbors2Depth(vector<BBTrk> targets, vector<RectID> occ_targets, int parent_occ_group_min_id) {

	int min_id = parent_occ_group_min_id;
	vector<RectID>::iterator iterR;
	for (iterR = occ_targets.begin(); iterR != occ_targets.end(); ++iterR) {

		vector<BBTrk>::iterator iterT;
		for (iterT = targets.begin(); iterT != targets.end(); ++iterT) {

			if (iterR->nid == iterT->id) {
				if (min_id > iterT->occTargets[0].min_id) { // when mininmum IDs in occlusion group are different
					
					min_id = iterT->occTargets[0].min_id;
				}
			}
		}
	}
	return min_id;
}
void SYM_MOT_HGMPHD::PredictFrmWise(int iFrmCnt, vector<BBTrk>& stats, const cv::Mat F, const cv::Mat Q, cv::Mat &Ps, int iPredictionLevel)
{
	int dims_state = stats.at(0).cov.cols;
	int dims_obs = stats.at(0).cov.cols - 2;

	if (iPredictionLevel == PREDICTION_LEVEL_LOW) {			// low level prediction

		vector<BBTrk>::iterator iter;

		for (iter = stats.begin(); iter < stats.end(); ++iter) {

			iter->fn = iFrmCnt;

			if (iPredictionLevel == PREDICTION_LEVEL_LOW) {
				if (dims_state == 4)
				{
					cv::Mat Ps_temp = Q_mid + F_mid*iter->cov*F_mid.t();

					// make covariance matrix diagonal
					//Ps_temp.copyTo(iter->cov); 
					iter->cov.at<double>(0, 0) = Ps_temp.at<double>(0, 0);
					iter->cov.at<double>(1, 1) = Ps_temp.at<double>(1, 1);
					iter->cov.at<double>(2, 2) = Ps_temp.at<double>(2, 2);
					iter->cov.at<double>(3, 3) = Ps_temp.at<double>(3, 3);

					//printf("%5.2lf %5.2lf %5.2lf %5.2lf\n", iter->cov.at<double>(0, 0), iter->cov.at<double>(0, 1), iter->cov.at<double>(0, 2), iter->cov.at<double>(0, 3));
					//printf("%5.2lf %5.2lf %5.2lf %5.2lf\n", iter->cov.at<double>(1, 0), iter->cov.at<double>(1, 1), iter->cov.at<double>(1, 2), iter->cov.at<double>(1, 3));
					//printf("%5.2lf %5.2lf %5.2lf %5.2lf\n", iter->cov.at<double>(2, 0), iter->cov.at<double>(2, 1), iter->cov.at<double>(2, 2), iter->cov.at<double>(2, 3));
					//printf("%5.2lf %5.2lf %5.2lf %5.2lf\n", iter->cov.at<double>(3, 0), iter->cov.at<double>(3, 1), iter->cov.at<double>(3, 2), iter->cov.at<double>(3, 3));
				}
				if (dims_state == 6)
				{
					cv::Mat Ps_temp = Q + F*iter->cov*F.t();
					Ps_temp.copyTo(iter->cov);
				}
			}
			//printf("\n");
			// copy current stat bounding box and velocity info to previous stat
			iter->vx_prev = iter->vx;
			iter->vy_prev = iter->vy;
			
			iter->rec_t_2 = iter->rec_t_1;
			iter->rec_t_1 = iter->rec;

			// 여기가 아니라 0.9*vt_1 + 0.1*vt 부분을 조정해야 하네 이 값의 abs가 1보다 작을경우 0.9, 1.0 을 조정한다.
			/*if (abs(iter->vx) >= 0.1) {
				float v = iter->vx;
				for (; abs(v) < 1; v += iter->vx);
				iter->vx = v;
				iter->rec.x += v;
			}
			if (abs(iter->vy) >= 0.1) {
				float v = iter->vy;
				for (; abs(v) < 1; v += iter->vy);
				iter->vy = v;
				iter->rec.y += v;
			}*/
			iter->rec.x += iter->vx; // 이거 왜 뺐었지?
			iter->rec.y += iter->vy;
			//printf("ID%d(%f,%f) ", iter->id,iter->vx, iter->vy);
		}
	}
}
void SYM_MOT_HGMPHD::QuadraticMotionEstimation(const vector<BBTrk>& stats) {

	vector<vector<Point>> stats_pts; // for parallel processing

									 // Access Map by stats' ID
	vector<BBTrk>::const_iterator iterS;
	for (iterS = stats.begin(); iterS != stats.end(); iterS++)
	{
		vector<Point> pts;
		vector<BBTrk>::iterator iterT;
		for (iterT = tracks_reliable[iterS->id].begin(); iterT != tracks_reliable[iterS->id].end(); iterT++) {
			pts.push_back(cv::Point(iterT->rec.x + iterT->rec.width / 2.0, iterT->rec.x + iterT->rec.height / 2.0));
		}
		stats_pts.push_back(pts);
	}

	// 
}
cv::Point2f SYM_MOT_HGMPHD::LinearMotionEstimation(map<int, vector<BBTrk>> tracks, int id) {

	float fl = tracks[id].back().fn - tracks[id].front().fn;

	cv::Rect r1, r2;
	cv::Point2f cp1, cp2;
	cv::Point2f v;

	r1 = tracks[id].front().rec;
	r2 = tracks[id].back().rec;

	cp1 = cv::Point2f(r1.x + r1.width / 2.0, r1.y + r1.height / 2.0);
	cp2 = cv::Point2f(r2.x + r2.width / 2.0, r2.y + r2.height / 2.0);

	v.x = (cp2.x - cp1.x) / fl;
	v.y = (cp2.y - cp1.y) / fl;

	//printf("ID%d(%d:%d,%d,%d,%d)(%.2lf,%.2lf)-(%d:%d,%d,%d,%d)(%.2lf,%.2lf)[%.2lf,%.2lf]\n", id, \
		tracks[id].front().fn, r1.x, r1.y, r1.width, r1.height, cp1.x, cp1.y,\
		tracks[id].back().fn, r2.x, r2.y, r2.width, r2.height, cp2.x, cp2.y,\
		v.x, v.y);


	return v;
}
bool SYM_MOT_HGMPHD::IsOutOfFrame(int x, int y, int w, int h, int fWidth, int fHeight) {
	// 중심점 기준 out of frame
	//int xc = x + w / 2; int yc = y + h / 2;
	////if (/*x < 0 || y < 0 || x >= fWidth || y >= fHeight || x + w >= fWidth || y + h >= fHeight*/) return true;
	//if (xc < 0 || yc < 0 || xc >= fWidth || yc >= fHeight ) return true;
	//else return false;

	// frame 밖에 있는 면적기준 out of frame
	cv::Rect obj(x, y, w, h);
	cv::Rect frm(0, 0, fWidth, fHeight);
	if (x < 0) { frm.x -= x; obj.x -= x; }
	if (y < 0) { frm.y -= y; obj.y -= y; }

	if ((obj&frm).area() < obj.area() / 3) return true;
	else return false;
}
float SYM_MOT_HGMPHD::FrameWiseAffinity(BBDet ob, BBTrk &stat_temp, const int dims_obs) {

	// Bounding box size contraint
	if ((stat_temp.rec.area() >= ob.rec.area() * 2) || (stat_temp.rec.area() * 2 <= ob.rec.area())) return 0.0;

	// Bounding box location contraint(gating)
	if (stat_temp.rec.area() >= ob.rec.area()) {
		if ((stat_temp.rec & ob.rec).area() < ob.rec.area() / 2) return 0.0;
	}
	else {
		if ((stat_temp.rec & ob.rec).area() < stat_temp.rec.area() / 2) return 0.0;
	}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;
	int dims_stat = dims_obs + 2;
	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	// (23)
	cv::Mat z_cov_temp(dims_obs, dims_obs, CV_64FC1);
	//z_cov_temp = H*Ps*H.t() + R;

	z_cov_temp = H_mid*stat_temp.cov*H_mid.t() + R_mid;

	//K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	K = stat_temp.cov*H_mid.t()*z_cov_temp.inv(DECOMP_SVD);
	// (22)
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);
	//Ps_temp = Ps - K*H*Ps;
	Ps_temp = stat_temp.cov - K*H_mid*stat_temp.cov;
	Ps_temp.copyTo(stat_temp.cov);

	cv::Mat z_temp(dims_obs, 1, CV_64FC1);
	z_temp = (Mat_<double>(dims_obs, 1) << ob.rec.x + (double)ob.rec.width / 2.0, ob.rec.y + (double)ob.rec.height / 2.0/*, ob.rec.width, ob.rec.height*/);

	// (20)
	// H*GMM[i] : k-1 와 k-2 사이의 속도로 추론한 state를 Observation으로 transition
	// width 와 height에는 속도 미적용
	cv::Mat mean_obs(dims_obs, 1, CV_64FC1);
	mean_obs.at<double>(0, 0) = (double)stat_temp.rec.x + (double)stat_temp.rec.width / 2.0;
	mean_obs.at<double>(1, 0) = (double)stat_temp.rec.y + (double)stat_temp.rec.height / 2.0;

	q_value = this->GaussianFunc(dims_obs, z_temp, mean_obs, z_cov_temp);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
float SYM_MOT_HGMPHD::TrackletWiseAffinity(BBTrk &stat_pred, const BBTrk& obs, const int& dims_obs) {

	//printf("lost ID%d(%d,%d,%d,%d) to live ID%d(%d,%d,%d,%d)\n", \
		stat_pred.id, stat_pred.rec.x, stat_pred.rec.y, stat_pred.rec.width, stat_pred.rec.height, \
		obs.id, obs.rec.x, obs.rec.y, obs.rec.width, obs.rec.height);

	// Bounding box size contraint
	if ((stat_pred.rec.area() >= obs.rec.area() * 2) || (stat_pred.rec.area() * 2 <= obs.rec.area())) return 0.0;

	// Bounding box location contraint(gating)
	if (stat_pred.rec.area() >= obs.rec.area()) {
		if ((stat_pred.rec & obs.rec).area() <= 0 /*obs.rec.area() / 4*/) return 0.0;
	}
	else {
		if ((stat_pred.rec & obs.rec).area() <=0  /*stat_pred.rec.area() / 4*/) return 0.0;
	}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;
	int dims_stat = dims_obs + 2;
	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	// (23)
	cv::Mat z_cov_temp(dims_obs, dims_obs, CV_64FC1);
	//z_cov_temp = H*Ps*H.t() + R;

	z_cov_temp = H_mid*stat_pred.cov*H_mid.t() + R_mid;

	//K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	K = stat_pred.cov*H_mid.t()*z_cov_temp.inv(DECOMP_SVD);
	// (22)
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);
	//Ps_temp = Ps - K*H*Ps;
	Ps_temp = stat_pred.cov - K*H_mid*stat_pred.cov;
	Ps_temp.copyTo(stat_pred.cov);

	cv::Mat z_temp(dims_obs, 1, CV_64FC1);
	z_temp = (Mat_<double>(dims_obs, 1) << obs.rec.x + (double)obs.rec.width / 2.0, obs.rec.y + (double)obs.rec.height / 2.0/*, obs_live.vx, obs_live.vy*/);

	// (20)
	// H*GMM[i] : k-1 와 k-2 사이의 속도로 추론한 state를 Observation으로 transition
	// width 와 height에는 속도 미적용
	cv::Mat mean_obs(dims_obs, 1, CV_64FC1);
	mean_obs.at<double>(0, 0) = (double)stat_pred.rec.x + (double)stat_pred.rec.width / 2.0;
	mean_obs.at<double>(1, 0) = (double)stat_pred.rec.y + (double)stat_pred.rec.height / 2.0;
	//mean_obs.at<double>(2, 0) = (double)stat_temp.vx;
	//mean_obs.at<double>(3, 0) = (double)stat_temp.vy;

	q_value = this->GaussianFunc(dims_obs, z_temp, mean_obs, z_cov_temp);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}float SYM_MOT_HGMPHD::TrackletWiseAffinityVelocity(BBTrk &stat_pred, const BBTrk& obs, const int& dims_obs) {

	//printf("lost ID%d(%d,%d,%d,%d) to live ID%d(%d,%d,%d,%d)\n", \
			stat_pred.id, stat_pred.rec.x, stat_pred.rec.y, stat_pred.rec.width, stat_pred.rec.height, \
		obs.id, obs.rec.x, obs.rec.y, obs.rec.width, obs.rec.height);

// Bounding box size contraint
	if ((stat_pred.rec.area() >= obs.rec.area() * 2) || (stat_pred.rec.area() * 2 <= obs.rec.area())) return 0.0;

	// Bounding box location contraint(gating)
	//if (stat_pred.rec.area() >= obs.rec.area()) {
	//if ((stat_pred.rec & obs.rec).area() < obs.rec.area() / 2) return 0.0;
	//}
	//else {
	//if ((stat_pred.rec & obs.rec).area() < stat_pred.rec.area() / 2) return 0.0;
	//}

	// Step2: Update each Gaussian for every observation
	// find the observation which makes the Gaussian's weight into maximum among every observation

	// Step 2: Update phase1
	double q_value = 0.0;
	int dims_stat = dims_obs + 2;
	cv::Mat K(dims_stat, dims_obs, CV_64FC1);

	// (23)
	cv::Mat z_cov_temp(dims_obs, dims_obs, CV_64FC1);
	//z_cov_temp = H*Ps*H.t() + R;

	z_cov_temp = H_mid*stat_pred.cov*H_mid.t() + R_mid;

	//K = Ps*H.t()*z_cov_temp.inv(DECOMP_SVD);
	K = stat_pred.cov*H_mid.t()*z_cov_temp.inv(DECOMP_SVD);
	// (22)
	cv::Mat Ps_temp(dims_stat, dims_stat, CV_64FC1);
	//Ps_temp = Ps - K*H*Ps;
	Ps_temp = stat_pred.cov - K*H_mid*stat_pred.cov;
	Ps_temp.copyTo(stat_pred.cov);

	cv::Mat z_temp(dims_obs, 1, CV_64FC1);
	z_temp = (Mat_<double>(dims_obs, 1) << obs.rec.x + (double)obs.rec.width / 2.0, obs.rec.y + (double)obs.rec.height / 2.0, obs.vx, obs.vy);

	// (20)
	// H*GMM[i] : k-1 와 k-2 사이의 속도로 추론한 state를 Observation으로 transition
	// width 와 height에는 속도 미적용
	cv::Mat mean_obs(dims_obs, 1, CV_64FC1);
	mean_obs.at<double>(0, 0) = (double)stat_pred.rec.x + (double)stat_pred.rec.width / 2.0;
	mean_obs.at<double>(1, 0) = (double)stat_pred.rec.y + (double)stat_pred.rec.height / 2.0;
	mean_obs.at<double>(2, 0) = (double)stat_pred.vx;
	mean_obs.at<double>(3, 0) = (double)stat_pred.vy;

	q_value = this->GaussianFunc(dims_obs, z_temp, mean_obs, z_cov_temp);

	if (q_value < FLT_MIN) q_value = 0.0;
	return q_value;
}
void SYM_MOT_HGMPHD::DataAssocFrmWise(int iFrmCnt, const cv::Mat& img, vector<BBTrk>& stats, vector<BBDet>& obss, cv::Mat &Ps, const cv::Mat& H, double P_survive, int offset, int dims_low)
{
	//cv::Mat img_proc = img.clone();
	if (DEBUG_PRINT) printf("(a)");
	int min_cost = 0;
	int* m_cost = new int[obss.size()*stats.size()];
	int nObs = obss.size();
	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(stats.size(), 0.0));
	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(obss.size(), std::vector<BBTrk>(stats.size(), BBTrk()));
	for (int r = 0; r < obss.size(); ++r) {
		//stats.assign(stats_matrix[r].begin(), stats_matrix[r].end());
		for (int c = 0; c < stats.size(); ++c)
		{
			stats.at(c).CopyTo(stats_matrix[r][c]);
		}
	}
	if (DEBUG_PRINT) printf("(b)");
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < obs.size(); ++r){
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			// Calculate the Affinity between detection (observations) and tracking (states)
			double Taff = 1.0;
			q_values[r][c] = FrameWiseAffinity(obss[r], stats_matrix[r][c], 2);
			//printf("(%d,%d)%.20lf,%lf\n", r,c, stats_matrix[r][c].weight,q_values[r][c]);
			//if (!stats_matrix[r][c].isAlive) cout << "not alive(" << stats_matrix[r][c].id << ")" << endl;

			// float affinity = TrackAffinityLowSpatial(cmap, obs[r], stats[c]); // 이게 그냥 cov 값 높인다음 merge 하는것보다 성능이 좋으면!! 성공이다, 근데 속도가 ㅈㅈ네.. ㅜㅜ 0 아닌곳만 훑게하자

			if (q_values[r][c] < Taff* Q_TH_LOW) q_values[r][c] = 0.0;
		}
	}
	);
	if (DEBUG_PRINT) printf("(c)");
	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	//vector<vector<double>> weights;
	//weights.resize(nObs, std::vector<double>(stats.size(), 0.0));
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < obs.size(); ++r){
		int nStats = stats_matrix[r].size();
		// (19)
		double denominator = 0.0;													// (19)'s denominator(분모)
		for (int l = 0; l < stats_matrix[r].size(); ++l) {

			denominator += (stats_matrix[r][l].weight * q_values[r][l]);
			//if (q_values[r][l] > 0) printf("[obs%d, ID%d] %lf(from %lf)\n", r, stats_matrix[r][l].id,stats_matrix[r][l].weight, stats.at(l).weight);
		}
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			double numerator =  /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];	// (19)'s numerator(분자)
			stats_matrix[r][c].weight = numerator / denominator; //-> 이건 greedy 한 방법으로 해보자
			//stats_matrix[r][c].weight = q_values[r][c]; // hungarian 을 위함

			// Scaling the affinity value to Integer
			//printf("[obs%d]%.10lf(%.10lf/%.10lf, q:%.10lf) \n", r, stats_matrix[r][c].weight, numerator, denominator, q_values[r][c]);
			if (stats_matrix[r][c].weight > 0.0) {
				if (((double)(-1000000)*stats_matrix[r][c].weight /** 100000*/) < (double)(INT_MIN)) {
					std::cerr << "-1000000 * weight < INT_MIN" << std::endl;
					m_cost[r*nStats + c] = INT_MIN;
				}
				else {
					//printf("(%d,%d)%.20lf,%lf\n", r, c, stats_matrix[r][c].weight, q_values[r][c]);
					m_cost[r*nStats + c] = (int)((float)(-1000000)* (float)stats_matrix[r][c].weight/* * 100000*/);
				}
			}
			else {
				m_cost[r*nStats + c] = 0;
			}

			if (min_cost > m_cost[r*nStats + c])
				min_cost = m_cost[r*nStats + c];
		}
	}
	);
	if (DEBUG_PRINT) printf("(d)");
	for (int r = 0; r < obss.size(); ++r) {
		for (int c = 0; c < stats.size(); ++c) {
			//printf("%d ", m_cost[r*stats.size() + c]);
			m_cost[r*stats.size() + c] = m_cost[r*stats.size() + c] - min_cost + 1;
		}
		//printf("\n");
	}
	int max_cost = 1 - min_cost;

	if (DEBUG_PRINT) {
		printf("\n[Frame-wise Association]\n");
		this->PrintCostMatrix(m_cost, stats.size(), obss.size(), max_cost);
	}


	// Hungarian Method for solving data association problem (find the max cost assignment pairs)
	std::vector<vector<int>> assigns;
	assigns = this->HungarianMethod(m_cost, obss.size(), stats.size(), max_cost);
	//assigns = this->GreedyAssignMinCostPairs(m_cost, obss.size(), stats.size());

	bool *isAssignedStats = new bool[stats.size()];	memset(isAssignedStats, 0, stats.size());
	bool *isAssignedObs = new bool[obss.size()];	memset(isAssignedObs, 0, obss.size());
	int *isAssignedObsIDs = new int[stats.size()];	memset(isAssignedObsIDs, 0, stats.size()); // only used in LB_ASSOCIATION

	for (int c = 0; c < stats.size(); ++c) {
		for (int r = 0; r < obss.size(); ++r) {
			if (assigns[r][c] == 1 && m_cost[r*stats.size() + c] < max_cost) {

				// Velocity Update
				float vx_t_1 = stats[c].vx;
				float vy_t_1 = stats[c].vy;

				float vx_t = (obss[r].rec.x + obss[r].rec.width / 2.0) - (stats[c].rec.x + stats[c].rec.width / 2.0);
				float vy_t = (obss[r].rec.y + obss[r].rec.height / 2.0) - (stats[c].rec.y + stats[c].rec.height / 2.0);

				/*if (iFrmCnt > 2) {
				stats[c].vx = vx_t_1*0.9 + vx_t*0.1;
				stats[c].vy = vy_t_1*0.9 + vy_t*0.1;
				}
				else {*/
				stats[c].vx = vx_t_1*VELOCITY_UPDATE_ALPHA + vx_t*(1.0 - VELOCITY_UPDATE_ALPHA);
				stats[c].vy = vy_t_1*VELOCITY_UPDATE_ALPHA + vy_t*(1.0 - VELOCITY_UPDATE_ALPHA);
				//}
				// Bounding box Update
				stats[c].rec = obss[r].rec; // (x, y, width, height)
				stats[c].weight = stats_matrix[r][c].weight;
				//stats[c].rec.width = (double)(obs[r].rec.width)*0.1 + (double)(obs[r].rec.width)*0.9;
				//stats[c].rec.height = (double)(obs[r].rec.height)*0.1 + (double)(obs[r].rec.height)*0.9;

				// Covariance Matrix Update
				stats_matrix[r][c].cov.copyTo(stats[c].cov);

				// weight thresholding 과 normalization 과정 또한 필요 -> weight threholding 은 visual tracking 의 경우 1.0, 0 수준으로 나뉘어서 소용없다.

				cv::Rect tmplRect;
				tmplRect = RectExceptionHandling(img.cols, img.rows, tmplRect);
				if (tmplRect.width * tmplRect.height >= OBJ_MIN_SIZE) {
					cv::Mat tTmpl;
					CropRegularSizeObj(img, tmplRect, tTmpl);
					stats[c].tmpl = tTmpl.clone();
					tTmpl.release();
				}

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;
				isAssignedObsIDs[c] = obss[r].id; // only used in LB_ASSOCIATION
				break;
			}
			isAssignedStats[c] = false;
		}
		//if (cmap.empty())
		stats[c].isAlive = isAssignedStats[c];
		//else {
		//	if(isAssignedStats[c])	stats[c].weight *= 0.9; // PHD 필터대로 살려두니 쥐쥐네.. static 
		//	else stats[c].weight *= 0.1;
		//}
	}

	if (DEBUG_PRINT) {
		printf("\nID_t-1|ID_t");

		for (int c = 0; c < stats.size(); ++c) {
			printf("%d(%d) ", stats.at(c).id, isAssignedStats[c]);
		}
		printf("\n");
		for (int r = 0; r < obss.size(); ++r) {
			printf("%3d:\t", r);

			for (int c = 0; c < stats.size(); ++c) {
				printf("%3d ", assigns[r][c]);
			}
			printf("(%d)\n", isAssignedObs[r]);
		}
	}

	// Weight Normalization after GMPHD association process
	double sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		//if (cmap.empty()) {
		//if (stats[c].isAlive) {
		sumWeight += stats[c].weight;
		//printf("ID%d(%.5lf) ", stats[c].id, stats[c].weight);
		//}
		//}
		/*else {
		if (stats[c].weight < 0.1) {
		stats[c].isAlive = false;
		}
		else {
		sumWeight += stats[c].weight;
		}
		}*/
	}
	//printf("->");
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;
			//printf("ID%d(%.5lf) ", stats[c].id, stats[c].weight);
		}
	}
	//printf("\n");

	if (DEBUG_PRINT) printf("(d)");

	vector<int> newTracks;
	for (int r = 0; r < obss.size(); ++r) {
		if (!isAssignedObs[r]) {
			newTracks.push_back(r);
			BBTrk newTrk;
			newTrk.fn = iFrmCnt;
			newTrk.id = this->usedIDcnt++;
			newTrk.cov = (cv::Mat_<double>(4, 4) << \
				VAR_X, 0, 0, 0, \
				0, VAR_Y, 0, 0,
				0, 0, VAR_X_VEL, 0,
				0, 0, 0, VAR_Y_VEL);
			newTrk.rec = obss[r].rec;
			newTrk.isAlive = true;
			newTrk.vx = 0.0;
			newTrk.vy = 0.0;
			newTrk.weight = obss[r].weight;

			cv::Rect tmplRect;
			tmplRect = RectExceptionHandling(img.cols, img.rows, tmplRect);
			if (tmplRect.width * tmplRect.height >= OBJ_MIN_SIZE) {
				cv::Mat tTmpl;
				CropRegularSizeObj(img, tmplRect, tTmpl);
				newTrk.tmpl = tTmpl.clone();
				tTmpl.release();
			}

			stats.push_back(newTrk);
			if (DEBUG_PRINT) {
				printf("ID%d birth!!\n", newTrk.id);
			}
			if (LOG_PRINT) {
				FILE* fp;
				fp = fopen("res\\log\\log_gmphd.dat", "a");
				fprintf(fp, "ID%d birth!!\n", newTrk.id);
				fclose(fp);
			}
		}
	}
	if (DEBUG_PRINT) printf("(e)");
	if (MERGE_ON) {
		this->CheckOcclusionsMergeStates(stats);
	}

	// Weight Normalization After Birth Processing
	sumWeight = 0.0;
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			sumWeight += stats[c].weight;
			//printf("ID%d(%.3lf) ", stats[c].id, stats[c].weight);
		}
	}
	//printf("->");
	for (int c = 0; c < stats.size(); ++c) {
		if (stats[c].isAlive) {
			stats[c].weight /= sumWeight;
			//printf("ID%d(%.3lf) ", stats[c].id, stats[c].weight);
		}
	}
	//printf("\n");

	/// Memory Deallocation
	delete[]isAssignedStats;
	delete[]isAssignedObs;
	delete[]isAssignedObsIDs;
	delete[]m_cost;
}
void SYM_MOT_HGMPHD::DataAssocTrkWise(int iFrmCnt, cv::Mat& img, vector<BBTrk>& stats_lost, vector<BBTrk>& obss_live) {
	//cv::Mat img_proc = img.clone();
	//if (DEBUG_PRINT) printf("(a)");
	int min_cost = 0;
	double min_cost_dbl = DBL_MAX;
	int* m_cost = new int[obss_live.size()*stats_lost.size()];
	//double* m_cost_dbl = new double[obss_live.size()*stats_lost.size()];
	int nObs = obss_live.size();
	vector<vector<double>> q_values;
	q_values.resize(nObs, std::vector<double>(stats_lost.size(), 0.0));
	vector<vector<BBTrk>> stats_matrix; // It can boost parallel processing?
	stats_matrix.resize(obss_live.size(), std::vector<BBTrk>(stats_lost.size(), BBTrk()));

	if (DEBUG_PRINT) {
		printf("\n");
		printf("live tracklets:");
		for (int r = 0; r < obss_live.size(); ++r) {
			printf("ID%d, ", obss_live[r].id);
		}
		printf("\n");
		printf("lost tracklets:");
		for (int c = 0; c < stats_lost.size(); ++c) {
			printf("ID%d, ", stats_lost[c].id);
		}
		printf("\n");
		printf("\n");
	}
	for (int r = 0; r < obss_live.size(); ++r) {
		//stats.assign(stats_matrix[r].begin(), stats_matrix[r].end());
		for (int c = 0; c < stats_lost.size(); ++c)
		{
			stats_lost.at(c).CopyTo(stats_matrix[r][c]);
		}
	}
	//if (DEBUG_PRINT) printf("(b)");
	//Concurrency::parallel_for(0, nObs, [&](int r) {, 병렬 쓸다리 없다.
	for (int r = 0; r < obss_live.size(); ++r) {
		for (int c = 0; c < stats_matrix[r].size(); ++c) {

			int lostID = stats_matrix[r][c].id;
			int liveID = obss_live.at(r).id;

			if (DEBUG_PRINT)
			printf("ID %d(%d-%d) with ID %d(%d-%d)\n", \
				lostID, this->tracks_reliable[lostID].front().fn, this->tracks_reliable[lostID].back().fn, \
				liveID, this->tracks_reliable[liveID].front().fn, this->tracks_reliable[liveID].back().fn);

			float fd = this->tracks_reliable[liveID].front().fn - this->tracks_reliable[lostID].back().fn;

			double Taff = 1.0;
			if (fd > 0 && fd < TRACK_ASSOCIATION_TERM) { // ==0 일때는 occlusion 을 감안해야 할듯 일단 >0 으로 해보자

				// Linear Motion Estimation
				cv::Point2f v = this->LinearMotionEstimation(this->tracks_reliable, lostID);

				BBTrk stat_pred;
				this->tracks_reliable[lostID].back().CopyTo(stat_pred);

				stat_pred.vx = v.x;
				stat_pred.vy = v.y;

				//printf("(%lf,%lf)\n",v.x,v.y);

				stat_pred.rec.x = stat_pred.rec.x + stat_pred.vx*fd;
				stat_pred.rec.y = stat_pred.rec.y + stat_pred.vy*fd;

				//this->cvBoundingBox(img, obs_pred.rec, this->color_tab[obs_pred.id % 26], 3);

				q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinity(stat_pred, this->tracks_reliable[liveID].front(), 2);
				// q_values[r][c] = /*pow(0.9,fd)**/TrackletWiseAffinityVelocity(stat_pred, this->tracks_reliable[liveID].front(), 4);
				// 아니면 tracklet에서 처음-끝 을 가져와서 할지, 이게 맞는듯..?
			}
			else
				q_values[r][c] = 0;
			// Calculate the Affinity between detection (observations) and tracking (states)
			//printf("(%d,%d)%.20lf,%lf\n", r,c, stats_matrix[r][c].weight,q_values[r][c]);
			//if (!stats_matrix[r][c].isAlive) cout << "not alive(" << stats_matrix[r][c].id << ")" << endl;

			// float affinity = TrackAffinityLowSpatial(cmap, obs[r], stats[c]); // 이게 그냥 cov 값 높인다음 merge 하는것보다 성능이 좋으면!! 성공이다, 근데 속도가 ㅈㅈ네.. ㅜㅜ 0 아닌곳만 훑게하자

			if (q_values[r][c] < Taff* Q_TH_LOW) q_values[r][c] = 0.0;
		}
	}
	//);
	//if (DEBUG_PRINT) printf("(c)");
	// Calculate States' Weights by GMPHD filtering with q values
	// Then the Cost Matrix is filled with States's Weights to solve Assignment Problem.
	//vector<vector<double>> weights;
	//weights.resize(nObs, std::vector<double>(stats.size(), 0.0));
	Concurrency::parallel_for(0, nObs, [&](int r) {
		//for (int r = 0; r < obs.size(); ++r){
		int nStats = stats_matrix[r].size();
		// (19)
		double denominator = 0.0;													// (19)'s denominator(분모)
		for (int l = 0; l < stats_matrix[r].size(); ++l) {

			denominator += (stats_matrix[r][l].weight * q_values[r][l]);
			//if (q_values[r][l] > 0) printf("[obs%d, ID%d] %lf(from %lf)\n", r, stats_matrix[r][l].id,stats_matrix[r][l].weight, stats.at(l).weight);
		}
		for (int c = 0; c < stats_matrix[r].size(); ++c) {
			double numerator =  /*P_detection*/stats_matrix[r][c].weight*q_values[r][c];	// (19)'s numerator(분자)
			stats_matrix[r][c].weight = numerator / denominator; //-> 이건 greedy 한 방법으로 해보자
																 //stats_matrix[r][c].weight = q_values[r][c]; // hungarian 을 위함

																 // Scaling the affinity value to Integer
																 //printf("[obs%d]%.10lf(%.10lf/%.10lf, q:%.10lf) \n", r, stats_matrix[r][c].weight, numerator, denominator, q_values[r][c]);
			if (stats_matrix[r][c].weight > 0.0) {
				if (((double)(-1000000)*stats_matrix[r][c].weight /** 100000*/) < (double)(INT_MIN)) {
					std::cerr << "-1000000 * weight < INT_MIN" << std::endl;
					m_cost[r*nStats + c] = INT_MIN;
				}
				else {
					//printf("(%d,%d)%.20lf,%lf\n", r, c, stats_matrix[r][c].weight, q_values[r][c]);
					m_cost[r*nStats + c] = (int)((float)(-1000000)* (float)stats_matrix[r][c].weight/* * 100000*/);
					//if(q_values[r][c]>0)
						//m_cost_dbl[r*nStats + c] = -log(q_values[r][c]);
				}
			}
			else {
				m_cost[r*nStats + c] = 0;
				//m_cost_dbl[r*nStats + c] = DBL_MAX;
			}

			if (min_cost > m_cost[r*nStats + c])
				min_cost = m_cost[r*nStats + c];
			//if (min_cost_dbl > m_cost_dbl[r*nStats + c])
				//min_cost_dbl = m_cost_dbl[r*nStats + c];
		}
	}
	);
	if (DEBUG_PRINT) printf("(d)");
	for (int r = 0; r < obss_live.size(); ++r) {
		for (int c = 0; c < stats_lost.size(); ++c) {
			//		printf("%d ", m_cost[r*stats.size() + c]);
			m_cost[r*stats_lost.size() + c] = m_cost[r*stats_lost.size() + c] - min_cost + 1;
		}
		//	//printf("\n");
	}
	int max_cost = 1 - min_cost;
	//double max_cost_dbl = DBL_MAX;

	if (DEBUG_PRINT) {
		printf("[Tracklet Association](max:%d)\n", max_cost);
		this->PrintCostMatrix(m_cost, stats_lost.size(), obss_live.size(), max_cost);

		//printf("[Tracklet Association](max:%lf)\n", max_cost_dbl);
		//this->PrintCostMatrix(m_cost, stats_lost.size(), obss_live.size(), max_cost_dbl);
	}


	// Hungarian Method for solving data association problem (find the max cost assignment pairs)
	std::vector<vector<int>> assigns;
	assigns = this->HungarianMethod(m_cost, obss_live.size(), stats_lost.size(), max_cost);
	//assigns = this->HungarianMethod(m_cost_dbl, obss_live.size(), stats_lost.size(), max_cost_dbl);
	//assigns = this->GreedyAssignMinCostPairs(m_cost, obss.size(), stats.size());

	bool *isAssignedStats = new bool[stats_lost.size()];	memset(isAssignedStats, 0, stats_lost.size());
	bool *isAssignedObs = new bool[obss_live.size()];	memset(isAssignedObs, 0, obss_live.size());

	for (int r = 0; r < obss_live.size(); ++r) {
		obss_live[r].id_associated = -1; // faild to tracklet association
		for (int c = 0; c < stats_lost.size(); ++c) {
			if (assigns[r][c] == 1 && m_cost[r*stats_lost.size() + c] < max_cost) {

				// obss_live[r].id = stats_lost[c].id;
				obss_live[r].id_associated = stats_lost[c].id;

				isAssignedStats[c] = true;
				isAssignedObs[r] = true;

				stats_lost[c].isAlive = true;

				break;
			}
		}
	}

	if (DEBUG_PRINT) {
		printf("\nID_t-1|ID_t");

		for (int c = 0; c < stats_lost.size(); ++c) {
			printf("%d(%d) ", stats_lost.at(c).id, isAssignedStats[c]);
		}
		printf("\n");
		for (int r = 0; r < obss_live.size(); ++r) {
			printf("%3d:\t", obss_live.at(r).id);

			for (int c = 0; c < stats_lost.size(); ++c) {
				printf("%3d ", assigns[r][c]);
			}
			printf("(%d)\n", obss_live[r].id_associated);
		}
	}

	delete[]m_cost;
	//delete[]m_cost_dbl;
	delete[]isAssignedObs;
	delete[]isAssignedStats;
}
// Rect Region Correction for preventing out of frame
cv::Rect SYM_MOT_HGMPHD::RectExceptionHandling(int fWidth, int fHeight, cv::Rect rect) {

	if (rect.x < 0) {
		//rect.width += rect.x;
		rect.x = 0;
	}
	if (rect.width < 0) rect.width = 0;
	if (rect.x >= fWidth) rect.x = fWidth - 1;
	if (rect.width > fWidth) rect.width = fWidth;
	if (rect.x + rect.width > fWidth) rect.width = fWidth - rect.x;

	if (rect.y < 0) {
		//rect.height += rect.y;
		rect.y = 0;
	}
	if (rect.height < 0) rect.height = 0;
	if (rect.y >= fHeight) rect.y = fHeight - 1;
	if (rect.height > fHeight) rect.height = fHeight;
	if (rect.y + rect.height > fHeight) rect.height = fHeight - rect.y;

	return rect;
}
void SYM_MOT_HGMPHD::CropObjRegionRegularization(IplImage* frame_input, cv::Rect rec, cv::Mat &obj_tmpl, bool regularSize) {
	cv::Mat trk_mat = cvarrToMat(frame_input);
	cv::Mat obj = trk_mat(rec).clone();
	cv::Mat obj_resize;
	if (regularSize)
		cv::resize(obj, obj_resize, cv::Size(OBJ_REGULAR_WIDTH, OBJ_REGULAR_HEIGHT));
	else
		obj_resize = obj;
	obj_tmpl = obj_resize.clone();

	obj_resize.release();
	obj.release();
}
double SYM_MOT_HGMPHD::MatchingMethod(cv::Mat input_templ, cv::Mat cand, int cand_index, int match_method) {
	cv::Mat candResize;
	if (input_templ.rows >= cand.rows || input_templ.cols >= cand.cols)
		cv::resize(cand, candResize, cv::Size(input_templ.cols, input_templ.rows));
	else
		candResize = cand.clone();

	/// Source image to display
	cv::Mat result, display;
	display = candResize.clone();
	//cand.copyTo(display);

	/// Create the result matrix
	int result_cols = candResize.cols - input_templ.cols + 1;
	int result_rows = candResize.rows - input_templ.rows + 1;

	result.create(result_rows, result_cols, CV_32FC1);

	/// Do the Matching
	matchTemplate(candResize, input_templ, result, match_method);
	double matchVal = (double)result.at<float>(0, 0);

	display.release();
	result.release();
	candResize.release();

	return matchVal;
}
void SYM_MOT_HGMPHD::GetObjectHistogram(Mat &frame, Mat objectTmpl, Rect objectRegion, Mat& objectHist, Rect& LocalSearchRegion) {

	const int channels[] = { 0, 1 };
	const int histSize[] = { 64, 64 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };

	// Histogram in object region
	Mat objectROI = objectTmpl;
	calcHist(&objectROI, 1, channels, noArray(), objectHist, 2, histSize, ranges, true, false);

	// A priori color distribution with cumulative histogram ()
	Rect LSRec;
	LSRec.x = objectRegion.x - 1 * objectRegion.width;
	LSRec.y = objectRegion.y - objectRegion.height / 2;
	LSRec.width = 3 * objectRegion.width;
	LSRec.height = 2 * objectRegion.height;

	LSRec = RectExceptionHandling(frame.cols, frame.rows, LSRec);

	LocalSearchRegion = LSRec;

	Mat localSearchROI = frame(LSRec);
	Mat localHist;
	calcHist(&localSearchROI, 1, channels, noArray(), localHist, 2, histSize, ranges, true, true);

	// Boosting: Divide conditional probabilities in object area by a priori probabilities of colors
	for (int y = 0; y < objectHist.rows; y++) {
		for (int x = 0; x < objectHist.cols; x++) {
			objectHist.at<float>(y, x) /= localHist.at<float>(y, x);
		}
	}
	normalize(objectHist, objectHist, 0, 255, NORM_MINMAX);
}
void SYM_MOT_HGMPHD::BackProjection(const Mat &frame, const Mat &obj_hist, Mat &bp) {
	const int channels[] = { 0, 1 };
	float range[] = { 0, 256 };
	const float *ranges[] = { range, range };
	calcBackProject(&frame, 1, channels, obj_hist, bp, ranges);
}
void SYM_MOT_HGMPHD::SetSeqName(string seq) {
	this->seqName = seq;
}
string SYM_MOT_HGMPHD::GetSeqName() {
	return this->seqName;
}
void SYM_MOT_HGMPHD::SetDetName(string det) {
	this->detName = det;
}
string SYM_MOT_HGMPHD::GetDetName() {
	return this->detName;
}
/* Check the occlusion between object region and user setting occlusion region*/
bool SYM_MOT_HGMPHD::IsOccluded(int &occlusionCase, CvRect obj1, CvRect obj2, CvRect &occ)
{
	// Use the target's histogram at the t-1 th frame instead of it at t-th frame in the occludded regions
	// if traget.at(i).x ~~~  = tracklet.back().hist.copyTo(traget.at(i).hist), copy value만 하자 
	/*
	-the inner rectangle indicates occluded region
	-the numbers 1 through 9 mean the occlusion cases by locations
	1  │ 7 │  4
	┌─┼───┼─┐
	──┼─┼───┼─┼──
	3│ │ 9 │ │6
	──┼─┼───┼─┼──
	└─┼───┼─┘
	2  │ 8 │  5
	*/
	/* PETS09 (410, 100, 55, 290)
	-표지판영역 : 410, 204, 45, 40
	-표지판 및 기둥영역 : 410, 100, 55(465-410), 290(390-100) */
	/* GIST ICT 2015 S4 (0, 0, 1, 1)
	-따로 지정한 영역이 없다는 이야기
	*/
	// occ_rect = cvRect(0, 0, 1, 1);
	// 지금 이 logic은 obj가 1,3,2 / 7,9,8 / 2,6,5 / 1,7,4 / 3,9,6 / 2,8,5 같은 경우를 고려하지 않고있다.
	// 물론 obj와 occ를 바꿔서 생각하면 되지만, 함수를 obj, occ 바꿔서 두번씩 호출해줘야 된다는 불편함이 있다.
	// 하나의 tree로 모두 찾아 낼 수 있게 만들자.
	cv::Point src, dst;
	bool isOccluded = false;
	occlusionCase = -1;
	if (obj1.x < obj2.x) {

		dst = Point(MIN(obj1.x + obj1.width, obj2.x + obj2.width),
			MIN(obj1.y + obj1.height, obj2.y + obj2.height));

		if (obj1.y < obj2.y)
		{
			src = Point(obj2.x, obj2.y);
		}
		else if (obj1.y > obj2.y + obj2.height)
		{
			src = Point(obj2.x, obj1.y);
		}
		else {  // obj_rect.y >= occ_rect.y && obj_rect.y <= occ_rect.y + occ_rect.height
			src = Point(obj2.x, obj2.y + obj2.height);

			int temp = src.y;
			src.y = dst.y;
			dst.y = temp;
		}
	}
	else if (obj1.x >= obj2.x && obj1.x < obj2.x + obj2.width) {

		dst = Point(MIN(obj1.x + obj1.width, obj2.x + obj2.width),
			MIN(obj1.y + obj1.height, obj2.y + obj2.height));

		if (obj1.y < obj2.y)
		{
			src = Point(obj1.x, obj2.y);
		}
		else if (obj1.y > obj2.y + obj2.height)
		{
			src = Point(obj1.x, obj2.y);
		}
		else {
			src = Point(obj1.x, obj2.y + obj2.height);

			int temp = src.y;
			src.y = dst.y;
			dst.y = temp;
		}
	}
	else return false; // obj1.x > obj2.x + obj2.width

	if (dst.x - src.x > 0 && dst.y - src.y > 0) {
		// return true;
		if ((dst.x - src.x) * (dst.y - src.y) * 3 > obj1.width * obj1.height)	return true;
	}
	else																	return false;
}
// Write the Tracking Results in a file(*.dat)
void SYM_MOT_HGMPHD::WriteTrackingResults(int cam_num, int iFrame_count, int Nbbs, const double *bbs, const int column_elements, int sync_offset, string dbname, string tag, double ThDetConf) {

	if (this->db_type != DB_TYPE_UA_DETRAC) {
		FILE* fp_track = NULL;

		char filePath_track[256];
		//	if (sync_offset == 0)
		if (!dbname.compare("") && tag.compare(""))
			sprintf_s(filePath_track, 256, "res\\track_dat\\track_%s.txt", tag.c_str());
		else if (!dbname.compare("") && !tag.compare(""))
			sprintf_s(filePath_track, 256, "res\\track_dat\\track.txt");
		else if (dbname.compare("") && tag.compare(""))
			sprintf_s(filePath_track, 256, "res\\track_dat\\%s_%s.txt", dbname.c_str(), tag.c_str());
		else if (dbname.compare("") && !tag.compare(""))
			sprintf_s(filePath_track, 256, "res\\track_dat\\%s.txt", dbname.c_str());
		//	else
		//		sprintf(filePath_track, "Results\\track_dat\\S%d_track_crop%s.dat", cam_num, tag);

		if (iFrame_count == 0)
			fp_track = fopen(filePath_track, "w+");
		else
			fp_track = fopen(filePath_track, "a");
		if (!fp_track) {
			printf("[ERROR]File open (line:3574)%d,%d,%d(%s)\n%s\n", iFrame_count, Nbbs, sync_offset, tag.c_str(), filePath_track);
			//return;
		}
		else if (feof(fp_track)) {
			printf("[ERROR]End of file (line:3578)\n");
			fclose(fp_track);
			return;
		}


		int rows = column_elements;
		if (this->db_type == DB_TYPE_MOT_CHALLENGE) {
			for (int i = 0; i < Nbbs; i++) {
				fprintf(fp_track, "%d,%d,%.2lf,%.2lf,%.2lf,%.2lf,-1,-1,-1,-1\n", iFrame_count + sync_offset, (int)bbs[i * rows + 0], bbs[i * rows + 1], bbs[i * rows + 2], bbs[i * rows + 3], bbs[i * rows + 4]);
			}
		}
		else if (this->db_type == DB_TYPE_ICT2) {
			if (Nbbs == 0)
				fprintf(fp_track, "%d %d 0 0 0 0 0 0\n", cam_num, iFrame_count + sync_offset);
			else {
				for (int i = 0; i < Nbbs; i++) {
					// ICT 2세부 "cam number, frame number, num of objects, id, x, y, width, height
					fprintf(fp_track, "%d %d %d %d %d %d %d %d\n", cam_num, iFrame_count + sync_offset, Nbbs, (int)bbs[i * rows + 0], (int)bbs[i * rows + 1], (int)bbs[i * rows + 2], (int)bbs[i * rows + 3], (int)bbs[i * rows + 4]);
				}
			}
		}
		fclose(fp_track);
	}
	else if (this->db_type == DB_TYPE_UA_DETRAC) {
		


		FILE* fp_LX = NULL,*fp_LY = NULL,*fp_W = NULL,*fp_H = NULL;
		
		char filePaths[4][256];
		string strThDetConf = boost::str(boost::format("%.1f") % ThDetConf);
		sprintf_s(filePaths[0], 256, "res\\UA-DETRAC\\%s\\%s\\%s_LX.txt", this->detName,strThDetConf,this->seqName.c_str());
		sprintf_s(filePaths[1], 256, "res\\UA-DETRAC\\%s\\%s\\%s_LY.txt", this->detName, strThDetConf,this->seqName.c_str());
		sprintf_s(filePaths[2], 256, "res\\UA-DETRAC\\%s\\%s\\%s_W.txt", this->detName, strThDetConf, this->seqName.c_str());
		sprintf_s(filePaths[3], 256, "res\\UA-DETRAC\\%s\\%s\\%s_H.txt", this->detName, strThDetConf, this->seqName.c_str());
		
		//cout << filePaths[0];

		if (iFrame_count == 0) {
			fp_LX = fopen(filePaths[0], "w+");
			fp_LY = fopen(filePaths[1], "w+");
			fp_W = fopen(filePaths[2], "w+");
			fp_H = fopen(filePaths[3], "w+");
		}
		else{
			fp_LX = fopen(filePaths[0], "a");
			fp_LY = fopen(filePaths[1], "a");
			fp_W = fopen(filePaths[2], "a");
			fp_H = fopen(filePaths[3], "a");
		}

		if (Nbbs == 0) {
			fprintf(fp_LX, "0\n");
			fprintf(fp_LY, "0\n");
			fprintf(fp_W, "0\n");
			fprintf(fp_H, "0\n");
		}
		else {
			for (int i = 0; i < Nbbs; i++) {
				fprintf(fp_LX, "%d,", (int)bbs[i * column_elements + 1]+1);
				fprintf(fp_LY, "%d,", (int)bbs[i * column_elements + 2]+1);
				fprintf(fp_W, "%d,", (int)bbs[i * column_elements + 3]);
				fprintf(fp_H, "%d,", (int)bbs[i * column_elements + 4]);
			}
			fprintf(fp_LX, "0\n");
			fprintf(fp_LY, "0\n");
			fprintf(fp_W, "0\n");
			fprintf(fp_H, "0\n");
		}
		fclose(fp_LX);
		fclose(fp_LY);
		fclose(fp_W);
		fclose(fp_H);
	}
}
/* lt : left top point, rb : right bottom point */
void SYM_MOT_HGMPHD::cvBoundingBox(IplImage* img, CvPoint lt, CvPoint rb, CvScalar color, int thick, int id, string type)
{
	// Font Initializtion
	if (id >= 0) {
		string strID = type.substr(0, 1) + to_string(id);

		cv::Point pt;
		pt.x = lt.x;
		if ((rb.y) < img->height / 2) pt.y = rb.y + 15;
		else							pt.y = lt.y - 5;;

		cv::putText(cv::cvarrToMat(img), strID, pt, FONT_HERSHEY_SIMPLEX, 0.5, color, 1);

		//cvPutText(img, carrID, cvPoint(lt.x, lt.y - 5), &font, color);	
		if (type.compare("")) {
			char carrType[16];
			sprintf_s(carrType, 16, "%s\0", type.c_str());

			//cvRectangle(img, cvPoint(lt.x, rb.y), cvPoint(rb.x, rb.y + 20), cvScalarAll(50), -1);
			//cv::putText(cv::cvarrToMat(img), carrType, cv::Point(lt.x + 5, rb.y + 15), FONT_HERSHEY_SIMPLEX, 0.5, cvScalar(255, 255, 255), 1);

		}
	}

	cvRectangle(img, lt, rb, color, thick);
}
/* rec : rectangle */
void SYM_MOT_HGMPHD::cvBoundingBox(cv::Mat& img, cv::Rect rec, cv::Scalar color, int thick, int id, string type)
{
	// Font Initializtion
	if (id >= 0) {
		string strID = type.substr(0, 1) + to_string(id);

		cv::Point pt;
		cv::Rect bg;
		pt.x = rec.x;
		if ((rec.y + rec.height) < img.rows / 2) { 
			pt.y = rec.y + rec.height + 15; 
			bg = cv::Rect(rec.x, rec.y + rec.height, 50, 25);

		}
		else {
			pt.y = rec.y - 5;
			bg = cv::Rect(rec.x, rec.y - 25, 50, 25);
		}

		cv::rectangle(img, bg, cv::Scalar(50, 50, 50), -1);
		cv::putText(img, strID, pt, FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0,255,255)/*color*/, 1);

		if (type.compare("")) {
			char carrType[16];
			sprintf_s(carrType, 16, "%s\0", type.c_str());
		}
	}

	cv::rectangle(img, rec, color, thick);
}
void SYM_MOT_HGMPHD::cvPrintMat(cv::Mat matrix, string name)
{
	/*
	<Mat::type()>
	depth에 channels까지 포함하는 개념 ex. CV_64FC1
	<Mat::depth()>
	CV_8U - 8-bit unsigned integers ( 0..255 )
	CV_8S - 8-bit signed integers ( -128..127 )
	CV_16U - 16-bit unsigned integers ( 0..65535 )
	CV_16S - 16-bit signed integers ( -32768..32767 )
	CV_32S - 32-bit signed integers ( -2147483648..2147483647 )
	CV_32F - 32-bit floating-point numbers ( -FLT_MAX..FLT_MAX, INF, NAN )
	CV_64F - 64-bit floating-point numbers ( -DBL_MAX..DBL_MAX, INF, NAN )
	*/
	printf("Matrix %s=\n", name);
	if (matrix.depth() == CV_64F) {
		//int channels = matrix.channels();
		for (int r = 0; r < matrix.rows; r++) {
			for (int c = 0; c < matrix.cols; c++) {
				//printf("(");
				//for( int cn=0 ; cn<channels ; cn++){
				printf("%6.2lf ", matrix.at<double>(r, c)/*[cn]*/);
				//} printf(")");
			}
			printf("\n");
		}
	}

}
void SYM_MOT_HGMPHD::CropRegularSizeObj(const cv::Mat& img, cv::Rect rec, cv::Mat &obj_tmpl, bool regularSize) {
	cv::Mat trk_mat = img;
	cv::Mat obj = trk_mat(rec).clone();
	cv::Mat obj_resize;
	if (regularSize)
		cv::resize(obj, obj_resize, cv::Size(OBJ_REGULAR_WIDTH, OBJ_REGULAR_HEIGHT));
	else
		obj_resize = obj;

	obj_tmpl = obj_resize.clone();

	obj_resize.release();
}
void SYM_MOT_HGMPHD::PrintCostMatrix(int *C, int nStats, int mObss, int max_cost) {

	int i, j;
	fprintf(stderr, "\n");

	for (i = 0; i < mObss; i++)
	{
		fprintf(stderr, " [");

		for (j = 0; j < nStats; j++)
		{
			if (C[i*nStats + j] == max_cost)
				fprintf(stderr, "-- ");
			else
				fprintf(stderr, "%5d ", C[i*nStats + j]);
		}
		fprintf(stderr, "]\n");
	}
	fprintf(stderr, "\n");

	if (LOG_PRINT_COST) {
		FILE* fp;
		fp = fopen("res\\log\\log.dat", "a");

		fprintf(fp, "\n");

		for (i = 0; i < mObss; i++)
		{
			fprintf(fp, " [");

			for (j = 0; j < nStats; j++)
			{
				if (C[i*nStats + j] == max_cost)
					fprintf(fp, "-- ");
				else
					fprintf(fp, "%5d ", C[i*nStats + j]);
			}
			fprintf(fp, "]\n");
		}
		fprintf(fp, "\n");

		fclose(fp);
	}
}
void SYM_MOT_HGMPHD::PrintCostMatrix(double *C, int nStats, int mObss, double max_cost) {

	int i, j;
	fprintf(stderr, "\n");

	for (i = 0; i < mObss; i++)
	{
		fprintf(stderr, " [");

		for (j = 0; j < nStats; j++)
		{
			if (C[i*nStats + j] == max_cost)
				fprintf(stderr, "-- ");
			else
				fprintf(stderr, "%.5lf ", C[i*nStats + j]);
		}
		fprintf(stderr, "]\n");
	}
	fprintf(stderr, "\n");

	if (LOG_PRINT_COST) {
		FILE* fp;
		fp = fopen("res\\log\\log.dat", "a");

		fprintf(fp, "\n");

		for (i = 0; i < mObss; i++)
		{
			fprintf(fp, " [");

			for (j = 0; j < nStats; j++)
			{
				if (C[i*nStats + j] == max_cost)
					fprintf(fp, "-- ");
				else
					fprintf(fp, "%.5lf ", C[i*nStats + j]);
			}
			fprintf(fp, "]\n");
		}
		fprintf(fp, "\n");

		fclose(fp);
	}
}
vector<vector<int>> SYM_MOT_HGMPHD::HungarianMethod(int* r, int nObs, int nStats, int min_cost) {

	std::vector< std::vector<double> > costMatrix = array_to_matrix_dbl(r, nObs, nStats, min_cost);

	HungarianAlgorithm HungAlgo;
	vector<vector<int>> assigns;
	assigns.resize(nObs, std::vector<int>(nStats, 0));

	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++) {
		//if (HUNGARIAN_PRINT) std::cout << x << "," << assignment[x] << "\t";
		if (assignment[x] >= 0)
			assigns[assignment[x]][x] = 1;
	}
	//if (HUNGARIAN_PRINT) printf("{end}");
	//std::cout << "\ncost: " << cost << std::endl;
	return assigns;
}
vector<vector<int>> SYM_MOT_HGMPHD::HungarianMethod(double* r, int nObs, int nStats, double min_cost) {

	std::vector< std::vector<double> > costMatrix = array_to_matrix_dbl(r, nObs, nStats, min_cost);

	HungarianAlgorithm HungAlgo;
	vector<vector<int>> assigns;
	assigns.resize(nObs, std::vector<int>(nStats, 0));

	vector<int> assignment;

	double cost = HungAlgo.Solve(costMatrix, assignment);

	for (unsigned int x = 0; x < costMatrix.size(); x++) {
		//if (HUNGARIAN_PRINT) std::cout << x << "," << assignment[x] << "\t";
		if (assignment[x] >= 0)
			assigns[assignment[x]][x] = 1;
	}
	//if (HUNGARIAN_PRINT) printf("{end}");
	//std::cout << "\ncost: " << cost << std::endl;
	return assigns;
}
vector< std::vector<double> >  SYM_MOT_HGMPHD::array_to_matrix_dbl(int* m, int rows, int cols, int min_cost) {
	int i, j;
	int rows_hung = cols;
	int cols_hung = rows;
	std::vector< std::vector<double> > r;
	r.resize(rows_hung, std::vector<double>(cols_hung, min_cost));

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) {
			r[j][i] = m[i*cols + j];
		}
	}
	return r;
}
vector< std::vector<double> >  SYM_MOT_HGMPHD::array_to_matrix_dbl(double* m, int rows, int cols, double min_cost) {
	int i, j;
	int rows_hung = cols;
	int cols_hung = rows;
	std::vector< std::vector<double> > r;
	r.resize(rows_hung, std::vector<double>(cols_hung, min_cost));

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) {
			r[j][i] = m[i*cols + j];
		}
	}
	return r;
}
vector<vector<int>> SYM_MOT_HGMPHD::GreedyAssignMinCostPairs(int* r, int nObs, int nStats) {

	vector<vector<int>> costMatrix = array_to_matrix_int(r, nObs, nStats);
	vector<vector<int>> assigns;
	assigns.resize(nObs, std::vector<int>(nStats, 0));
	for (int c = 0; c < nStats; c++) {
		int min_cost = INT_MAX;
		int min_cost_index = -1;
		for (int r = 0; r < nObs; r++) {

			if (min_cost > costMatrix[r][c]) {
				min_cost = costMatrix[r][c];
				min_cost_index = r;
			}
		}
		if (min_cost < 0 && min_cost_index >= 0)
		{
			assigns[min_cost_index][c] = 1;
			if (DEBUG_PRINT_COST) printf("%d(1) ", costMatrix[min_cost_index][c]);
		}
		else
			if (DEBUG_PRINT_COST) printf("0(0) ");
	}
	if (DEBUG_PRINT_COST) printf("\n");

	return assigns;
}
// not transpose
std::vector< std::vector<int> >  SYM_MOT_HGMPHD::array_to_matrix_int(int* m, int rows, int cols, bool SQUARE_MATRIX) {
	int i, j;
	std::vector< std::vector<int> > r;
	int sizes = 0;
	if (rows >= cols) {
		sizes = rows;
		r.resize(sizes, std::vector<int>(sizes, 0));
	}
	else {
		sizes = cols;
		r.resize(sizes, std::vector<int>(sizes, 0));
	}

	for (i = 0; i < rows; i++)
	{
		for (j = 0; j < cols; j++) {
			r[i][j] = m[i*cols + j];
		}
	}

	if (rows == cols || !SQUARE_MATRIX) return r;
	else {
		// Add dummy rows or columns for Hungarian Method

		if (sizes == rows) { // rows(observations) > cols(states)
			printf("Add %d more dummy columns\n", rows - cols);
			for (int i = 0; i < rows; i++)		// row iteration
			{
				for (j = cols; j < rows; j++) { // column iteration
					r[i][j] = 0;
				}
			}
		}
		else { // sizes == cols, rows(observations) < cols(states)
			printf("Add %d more dummy rows\n", cols - rows);
			for (i = rows; i < cols; i++)		// row iteration
			{
				for (j = 0; j < cols; j++) {	// column iteration
					r[i][j] = 0;
				}
			}
		}
		return r;
	}
}