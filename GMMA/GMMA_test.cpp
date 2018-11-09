// GMMA_test.cpp : 콘솔 응용 프로그램에 대한 진입점을 정의합니다.
//

#include "stdafx.h"

#define DEBUG_PRINT_MAIN 0

// The Function for Reading Detection Responses from txt
vector<string> ReadAllDetections(string detFilePath);
// The Function for Sorting Detection Responses by frame number (ascending)
vector<string> SortAllDetections(vector<string>& allLines);
// The Function for converting all string lines to double array (frame by frame)
double* StrVec2DblArray(int iFrameNum, vector<string>& allLines, vector<string>::iterator& iter, int &Nbbs, double resize_factor = 1.0, int db_type = DB_TYPE_MOT_CHALLENGE);
// The Function for Writing All Tracking Results in format of UA-DETRAC dataset.
void WriteAllTracks2Txt(vector<vector<BBTrk>> allTracks, string DetName, string strThDetConf, string SeqName, int DB_TYPE);
// The Function for Approximating the Lost Tracks which were recovered by T2TA.
void ApproximateAllTracks(vector<vector<BBTrk>> inputTracks, vector<vector<BBTrk>>& outputTracks);

int main()
{
	string detFilePath, videoFilePath, seqListFilePath;
	string imgPaths[30000];
	int totalFrames = 0, totalSequences = 1;
	double totalProcessingSecs = 0.0;

	int DB_TYPE = DB_TYPE_UA_DETRAC;
	/*----UA-DETRAC----*/
	string DETRAC_detName = "";
	vector<string> DETRAC_SeqsVec;
	string DETRAC_seqPaths[60];
	string DETRAC_detPaths[60];
	string DETRAC_train = "F://UA-DETRAC//train//";
	string DETRAC_test = "F://UA-DETRAC//test//";

	int detType = 1;
	cout << "Select a detector (txt) [1] CompACT, [2] R-CNN, [3] ACF, [4] DPM: ";
	fflush(stdin); cin >> detType;
	if (detType == 1) DETRAC_detName = "CompACT";
	else if (detType == 2) DETRAC_detName = "R-CNN";
	else if (detType == 3) DETRAC_detName = "ACF";
	else if (detType == 4) DETRAC_detName = "DPM";

	seqListFilePath = "sequences_UA-DETRAC_train_all.txt";// "sequences_UA-DETRAC_train_all.txt";
	DETRAC_SeqsVec = ReadAllDetections(string(seqListFilePath));
	totalSequences = DETRAC_SeqsVec.size();
	
	for (int sq = 0; sq < totalSequences; ++sq) {
		DETRAC_seqPaths[sq] = DETRAC_train + "img//" + DETRAC_SeqsVec[sq] + "//";
		DETRAC_detPaths[sq] = DETRAC_train + "det//" + DETRAC_detName + "//" + DETRAC_SeqsVec[sq] + "_Det_" + DETRAC_detName + ".txt";
		//cout << DETRAC_seqPaths[sq] << endl;
		//cout << DETRAC_detPaths[sq] << endl;
	}
	//cout << totalSequences << endl;

	for (int sq = 0; sq < totalSequences; ++sq) {
		clock_t end, start;
		totalProcessingSecs = 0.0;

		FILE* fp_detection = NULL;
		vector<string> allDetections;
		vector<string>::iterator iter_last;

		// UA-DETRAC
		allDetections = ReadAllDetections(DETRAC_detPaths[sq]); // a det.txt

		iter_last = allDetections.begin();
		totalFrames = 0;
		string imgPath;
		imgPath = DETRAC_seqPaths[sq];

		boost::filesystem::path p(imgPath.c_str());
		for (boost::filesystem::directory_iterator it(p); it != boost::filesystem::directory_iterator(); ++it) {
			imgPaths[totalFrames] = it->path().string();
			if (0) { // if a file does not have image format extension.
				exit(1);
			}
			else
				totalFrames++;
		}
		// SYM_MOT_HGMPHD LocalTracker;
		double Th_TA_Conf = 0.0; // minimum threshold of detection confidence 
		double Th_Confs[11] = { 0.0,0.1,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
		for (int cfs = 0; cfs < 10; cfs++) {
			totalProcessingSecs = 0.0;
			iter_last = allDetections.begin();
			int frames_skip_interval = 1;
			int iTotalTrackers = 1;
			SYM_MOT_HGMPHD tracker;

			tracker.db_type = DB_TYPE;

			tracker.trackObjType = OBJECT_TYPE_CAR;
			tracker.SetSeqName(DETRAC_SeqsVec[sq]);
			tracker.SetDetName(DETRAC_detName);

			int iFrameCnt;
			for (iFrameCnt = 0; ; iFrameCnt++) {

				cerr << "(" << iFrameCnt;
				cerr << "/";
				cerr << totalFrames;
				cerr << ") ";
				if (!DEBUG_PRINT) cerr << "\r";
				if (iFrameCnt >= totalFrames) {
					printf("All frames has been processed.\n");
					break;
				}
				cv::Mat img = cv::imread(string(imgPaths[iFrameCnt]));

				cv::Mat img_det = img.clone();	// Detection
				cv::Mat img_trk = img.clone();	// Tracking	at t
				cv::Mat img_trk_delay;			// Tracking at t-9

				int nObjects = 0;
				int nbs = 0;
				double* bbs = NULL;

				int nTrgs = 0;
				double* bbs_trk = NULL;

				double t = (double)getTickCount();

				CvScalar obj_colors[6];
				obj_colors[0] = cvScalar(255, 255, 255);
				obj_colors[1] = cvScalar(255, 0, 0);
				obj_colors[2] = cvScalar(0, 0, 0);
				obj_colors[3] = cvScalar(0, 0, 255);
				obj_colors[4] = cvScalar(0, 255, 255);
				obj_colors[5] = cvScalar(255, 255, 0);

				if (DEBUG_PRINT_MAIN) printf("[1]");


				double *bbs_det = NULL;
				double resize_factor = 1.0;

				bbs_det = StrVec2DblArray(iFrameCnt, allDetections, iter_last, nObjects, resize_factor, DB_TYPE);

				for (int i = 0; i < nObjects; i++)
				{
					nbs++;
				}

				if (nbs > 0) bbs = (double*)malloc(sizeof(double)*nbs * 5);
				int cnt = 0;

				for (int i = 0; i < nObjects; i++)
				{
					//int type = OBJECT_TYPE_CAR;
					int type = bbs_det[5 * nObjects + i];
					bbs[0 * nbs + cnt] = bbs_det[0 * nObjects + i];
					bbs[1 * nbs + cnt] = bbs_det[1 * nObjects + i];
					bbs[2 * nbs + cnt] = bbs_det[2 * nObjects + i];
					bbs[3 * nbs + cnt] = bbs_det[3 * nObjects + i];
					bbs[4 * nbs + cnt] = bbs_det[4 * nObjects + i];

					cnt++;
				}
				if (cvWaitKey(10) == 3) {
					if (bbs_det != NULL) free(bbs_det);
				}
				if (bbs_det != NULL) free(bbs_det);

				if (DEBUG_PRINT_MAIN) printf("[2]");


				// Do Tracking
				if (iFrameCnt % frames_skip_interval == 0) {
					//Concurrency::parallel_for(0, iTotalTrackers, [&](int iTrkers) {
					for (int iTrkers = 0; iTrkers < iTotalTrackers; iTrkers++) {

						if (iFrameCnt == 0) {
							tracker.SetTotalFrames(totalFrames);
						}
						bbs_trk = tracker.DoMOT(iFrameCnt / frames_skip_interval, img, nTrgs, bbs, nbs, Th_Confs[cfs]);
					}
					// Low & Mid level Tracking
					img_trk_delay = tracker.imgBatch[0].clone();
					//img_trk_delay = img_trk.clone();
					// Low level only
					//frame_track_delay = cvCloneImage(frame_track);
				}
				if (DEBUG_PRINT_MAIN) printf("[3]");

				// Calculating processing time including only tracking
				t = ((double)getTickCount() - t) / getTickFrequency();//elapsed time			
				totalProcessingSecs += t;

				// Drawing the Bounding Boxes of Detection and Tracking
				for (int iTrkers = 0; iTrkers < iTotalTrackers; iTrkers++) {

					// Detection Results
					for (int i = 0; i < nbs; i++) {

						double confidence = bbs[4 * nbs + i];

						if (confidence >= Th_Confs[cfs]) {
							int x = (int)bbs[0 * nbs + i];
							int y = (int)bbs[1 * nbs + i];
							int width = (int)bbs[2 * nbs + i];
							int height = (int)bbs[3 * nbs + i];

							int xc = x + width / 2;
							int yc = y + height / 2;
							// Draw Detection Bounding Boxes
							cv::rectangle(img_det, cvPoint(x, y), cvPoint(x + width, y + height), obj_colors[iTrkers], 2);

							// Convert double to char array
							std::ostringstream ost;
							ost << confidence;
							std::string str = ost.str();
							char cArrConfidence[8], cArrObsID[8];
							int c;
							for (c = 0; c < 5 && c < str.size(); c++) cArrConfidence[c] = str.c_str()[c];
							cArrConfidence[c] = '\0';
							cv::putText(img_det, cArrConfidence, cvPoint(x, y - 5), FONT_HERSHEY_SIMPLEX, 0.4, obj_colors[iTrkers], 1);

							// Draw Observation ID (not target ID)
							sprintf_s(cArrObsID, 8, "%d", i);
							cv::putText(img_det, cArrObsID, cvPoint(xc, yc), FONT_HERSHEY_SIMPLEX, 0.4, obj_colors[iTrkers], 1);
						}
					}
					// Tracking Results
					for (int i = 0; i < nTrgs; i++)
					{
						int id = (int)bbs_trk[i * 5 + 0];
						int x = (int)bbs_trk[i * 5 + 1];
						int y = (int)bbs_trk[i * 5 + 2];
						int width = (int)bbs_trk[i * 5 + 3];
						int height = (int)bbs_trk[i * 5 + 4];
						//int status = (int)bbs_trks[iTrkers][i * 6 + 5];

						// Draw tracking bounding boxes in tracking result image
						string objType = "";
						if (tracker.trackObjType == OBJECT_TYPE_PERSON)		objType = "Person";
						if (tracker.trackObjType == OBJECT_TYPE_CAR)		objType = "Car";
						if (tracker.trackObjType == OBJECT_TYPE_BICYCLE)	objType = "Bicyle";
						if (tracker.trackObjType == OBJECT_TYPE_SUITCASE)	objType = "Suitcase";
						if (tracker.trackObjType == OBJECT_TYPE_CHAIR)		objType = "Chair";
						if (tracker.trackObjType == OBJECT_TYPE_TRUCK)		objType = "Truck";

						//if (status > 0)
						tracker.cvBoundingBox(img_trk_delay, cv::Rect(x, y, width, height), \
							tracker.color_tab[id % (MAX_OBJECTS - 1)]/* obj_colors[iTrkers]*/, 2, id, objType);


						//if (id == 55 /*|| id == 49*/) cv::waitKey();
					}
				}
				if (DEBUG_PRINT_MAIN) printf("[4]");
				// Display Frame Number on Input & Tracking reseult image
				CvFont font;
				char* frameCntBuf = (char *)malloc(sizeof(char) * 8);
				sprintf_s(frameCntBuf, 8, "%d", (iFrameCnt) / frames_skip_interval);

				// Frame Count on Detection
				cv::putText(img_det, frameCntBuf, cvPoint(img.cols - 100, img.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 0), 2);
				// Frame Count on Tracking (low level)
				//cvPutText(frame_track, frameCntBuf, cvPoint(frame_track->width - 100, frame_track->height - 10), &font, cvScalar(0, 0, 0));
				// Frame Count on Tracking (mid level)
				sprintf_s(frameCntBuf, 8, "%d", (iFrameCnt - FRAMES_DELAY) /*/ frames_skip_interval*/);
				cv::putText(img_trk_delay, frameCntBuf, cvPoint(img.cols - 100, img.rows - 10), CV_FONT_HERSHEY_SIMPLEX, 1, cvScalar(0, 0, 0), 2);
				free(frameCntBuf);


				// Draw frames per second
				string text = format("%0.1f fps", 1.0 / t * frames_skip_interval);
				Scalar textColor(0, 0, 250);
				cv::putText(img_trk_delay, text, Point(10, 50), FONT_HERSHEY_PLAIN, 2, textColor, 2);

				// Diplay
				if (iFrameCnt % frames_skip_interval == 0) {
					//cv::imshow("Detection", img_det);
					//cv::imshow("Tracking", img_trk_delay);
				}
				// Save images
				if (iFrameCnt % frames_skip_interval == 0) {

					//gSaveAImage((iFrameCnt) / frames_skip_interval, frame_det, "res\\det_disp\\det", "jpg");
					//gSaveAImage(iFrameCnt / frames_skip_interval, frame_track, "res\\trk_disp\\trk1_", "jpg");
					//if (iFrameCnt >= FRAMES_DELAY) gSaveAImage((iFrameCnt - FRAMES_DELAY) / frames_skip_interval, &IplImage(img_trk_delay), "res\\trk_disp\\trk3_", "jpg");
					//gSaveAImage(iFrameCnt / frames_skip_interval, frame_det, "res\\det_disp\\det", "jpg");
				}

				if (DEBUG_PRINT_MAIN) printf("[5]");
				if (iFrameCnt == 0) {
					/*if (sq == 0) cvWaitKey();
					else*/ cvWaitKey(10);
					start = clock();
				}
				if (cvWaitKey(10) == 3) { // ctrl+c
					if (nbs > 0) if (bbs != NULL) free(bbs);

					if (iFrameCnt % frames_skip_interval == 0) {
						if (nTrgs > 0) if (bbs_trk != NULL) free(bbs_trk);
						img_trk_delay.release();
					}
					img.release();
					img_det.release();
					img_trk.release();
					break;
				}
				if (nbs > 0) if (bbs != NULL) free(bbs);
				if (iFrameCnt % frames_skip_interval == 0) {
					if (nTrgs > 0) if (bbs_trk != NULL) free(bbs_trk);
					img_trk_delay.release();
				}
				img.release();
				img_det.release();
				img_trk.release();
			}
			if (DEBUG_PRINT_MAIN) printf("[6]");


			end = clock() - start;
			int totalProcessingFrames = iFrameCnt / frames_skip_interval; // 눈에 보이는 속도가 아닌 처리속도
			printf("Processing time %.2lfs (%.2lffps)\n", (double)end / (double)1000., (double)totalProcessingFrames / (double)((double)end / (double)1000.0));
			printf("Processing time %.2lfs (%.2lffps) (tracking only)\n", totalProcessingSecs, 1.0 / (totalProcessingSecs / (double)totalProcessingFrames));



			if (DB_TYPE == DB_TYPE_UA_DETRAC) {
				char filePath[256];
				string strThDetConf = boost::str(boost::format("%.1f") % Th_Confs[cfs]);
				sprintf_s(filePath, 256, "res\\UA-DETRAC\\%s\\%s\\%s_Speed.txt", DETRAC_detName, strThDetConf, DETRAC_SeqsVec[sq], DB_TYPE);
				cout << filePath << endl;
				FILE* fp_Speed = fopen(filePath, "w+");
				fprintf(fp_Speed, "%.2lf\n", 1.0 / (totalProcessingSecs / (double)totalProcessingFrames));
				fclose(fp_Speed);

				// Write All Tracking Results to Text Files
				if (/*Th_Confs[cfs] >= Th_TA_Conf*/TRACK_APPROXIMIATION_ON) {
					// Track Approximation
					vector<vector<BBTrk>> allTracksAppr;
					ApproximateAllTracks(tracker.allLiveReliables, allTracksAppr);
					WriteAllTracks2Txt(allTracksAppr, DETRAC_detName, strThDetConf, DETRAC_SeqsVec[sq], DB_TYPE);
				}
				else {
					WriteAllTracks2Txt(tracker.allLiveReliables, DETRAC_detName, strThDetConf, DETRAC_SeqsVec[sq], DB_TYPE);
				}
			}

			// Release Memory of Tracker
			tracker.Destroy();


		}
	}

	return 0;
}
/* Read All Detections at Once*/
vector<string> ReadAllDetections(string detFilePath) {
	vector<string> allLines;
	ifstream infile(detFilePath);
	string line;
	while (getline(infile, line)) {
		allLines.push_back(line);
	}
	return allLines;
}
vector<string> SortAllDetections(vector<string>& allLines) {
	// ascending sort by frame number
	/// http://azza.tistory.com/entry/STL-vector-%EC%9D%98-%EC%A0%95%EB%A0%AC

	class T {
	public:
		int frameNum;
		string line;
		T(string s) {
			line = s;

			boost::char_separator<char> bTok(",");
			boost::tokenizer < boost::char_separator<char>>tokens(s, bTok);
			vector<string> vals;
			for (const auto& t : tokens)
			{
				vals.push_back(t);
			}
			frameNum = boost::lexical_cast<double>(vals.at(0));
		}
		bool operator<(const T &t) const {
			return (frameNum < t.frameNum);
		}
	};


	// Reconstruct the vector<T> from vector<string> for sorting
	vector<T> tempAllLines;
	vector<string>::iterator iter = allLines.begin();
	for (; iter != allLines.end(); iter++) {
		if (iter[0].size() < 4) break;

		tempAllLines.push_back(T(iter[0]));
	}
	// Sort the vector<T> by frame number
	std::sort(tempAllLines.begin(), tempAllLines.end());

	// Copy the sorted vector<T> to vector<string>
	vector<string> sortedAllLines;
	vector<T>::iterator iterT = tempAllLines.begin();
	for (; iterT != tempAllLines.end(); iterT++) {

		sortedAllLines.push_back(iterT[0].line);
	}
	std::sort(tempAllLines.begin(), tempAllLines.end());

	return sortedAllLines;
}
double* StrVec2DblArray(int iFrameNum, vector<string>& allLines, vector<string>::iterator& iter_last, int &Nbbs, double resize_factor, int db_type) {

	double* bbs = NULL;
	double bb[MAX_GAUSSIANS][6]; // x, y, width, height, confidence, object type

	char* tok = (char*)malloc(sizeof(char) * 2);
	if (db_type == DB_TYPE_MOT_CHALLENGE || db_type == DB_TYPE_UA_DETRAC) {
		strcpy(tok, ",");
		iFrameNum += 1; // frame offset, in the MOT challenge frame number start at 1
	}
	else if (db_type == DB_TYPE_ICT1 || db_type == DB_TYPE_ICT2) strcpy(tok, " ");

	int nObjects = 0;
	vector<string>::iterator iter;

	// allLines.end() works when iterator && vector are in the same range
	for (iter = iter_last; iter != allLines.end(); iter++) {
		if (iter[0].size() < 4) break; // iter!=allLines.end() is not working why?

		boost::char_separator<char> bTok(tok);
		boost::tokenizer < boost::char_separator<char>>tokens(iter[0], bTok);
		vector<string> vals;
		for (const auto& t : tokens)
		{
			vals.push_back(t);
		}
		if (db_type == DB_TYPE_MOT_CHALLENGE || db_type == DB_TYPE_ICT2) {
			/// MOT Challenge
			// [frame number, -1, x(top left), y(top left), width, height, confidence, -1, -1, -1]
			// int, -1, float, float, float, float, -1, -1, -1

			/// ICT2
			// [int int int int int int float]
			// frame number
			// number of objects
			// x (top left) 
			// y (top left)
			// width
			// height
			// confidence

			int frame_num_in_file = boost::lexical_cast<double>(vals.at(0));	// char * to int
			if (frame_num_in_file > iFrameNum) {
				//if (DEBUG_PRINT_MAIN) printf("%d!=%d\n", frame_num_in_file, iFrameNum);
				break;
			}

			bb[nObjects][0] = boost::lexical_cast<double>(vals.at(2));
			bb[nObjects][1] = boost::lexical_cast<double>(vals.at(3));
			bb[nObjects][2] = boost::lexical_cast<double>(vals.at(4));
			bb[nObjects][3] = boost::lexical_cast<double>(vals.at(5));
			bb[nObjects][4] = boost::lexical_cast<double>(vals.at(6));
			bb[nObjects][5] = OBJECT_TYPE_PERSON;

			if (DEBUG_PRINT_MAIN)
				printf("[%d] %.lf %.lf %.lf %.lf %.2lf %.lf\n", frame_num_in_file, bb[nObjects][0], bb[nObjects][1], bb[nObjects][2], bb[nObjects][3], bb[nObjects][4], bb[nObjects][5]);

			if ((iter + 1) == allLines.end()) break;
		}
		else if (db_type == DB_TYPE_ICT1) {
			//// ICT 1세부 detection 결과 load////
			/// ICT1
			// "frame number.jpg"
			// x (top left) 
			// y (top left)
			// x (bottom right)
			// y (bottom right)
			// object type{car, person, bicycle, suitcase, chair, truck}
			// confidence

			char fileName[32];
			int c = 0;
			for (; vals.at(0)[c] != '.'; c++) fileName[c] = vals.at(0)[c];
			fileName[c] = '\0';						// [frame_number]
			const char* result = _strdup(fileName);
			int frame_num_in_file = atoi(result);	// char * to int

													// if frame number(in file) is different with processing frame
			if (frame_num_in_file > iFrameNum) break;


			// x, y, width, height, confidence, object type
			if (vals.size() == 7) { // ex. object type: "person"
				bb[nObjects][0] = boost::lexical_cast<double>(vals.at(1));
				bb[nObjects][1] = boost::lexical_cast<double>(vals.at(2));
				bb[nObjects][2] = boost::lexical_cast<double>(vals.at(3)) - bb[nObjects][0];
				bb[nObjects][3] = boost::lexical_cast<double>(vals.at(4)) - bb[nObjects][1];
				bb[nObjects][4] = boost::lexical_cast<double>(vals.at(6));
			}
			if (vals.size() == 8) { // ex. object type: "tennis racket"
				bb[nObjects][0] = boost::lexical_cast<double>(vals.at(1));
				bb[nObjects][1] = boost::lexical_cast<double>(vals.at(2));
				bb[nObjects][2] = boost::lexical_cast<double>(vals.at(3)) - bb[nObjects][0];
				bb[nObjects][3] = boost::lexical_cast<double>(vals.at(4)) - bb[nObjects][1];
				bb[nObjects][4] = boost::lexical_cast<double>(vals.at(7));
			}

			string objType(vals.at(5));

			if (!objType.compare("person"))
				bb[nObjects][5] = OBJECT_TYPE_PERSON;
			else if (!objType.compare("car"))
				bb[nObjects][5] = OBJECT_TYPE_CAR;
			else if (!objType.compare("bicycle"))
				bb[nObjects][5] = OBJECT_TYPE_BICYCLE;
			else if (!objType.compare("suitcase"))
				bb[nObjects][5] = OBJECT_TYPE_SUITCASE;
			else if (!objType.compare("chair"))
				bb[nObjects][5] = OBJECT_TYPE_CHAIR;
			else if (!objType.compare("truck"))
				bb[nObjects][5] = OBJECT_TYPE_TRUCK;
			else {
				bb[nObjects][5] = -1.0;
				nObjects--;
			}
		}
		else if (db_type == DB_TYPE_UA_DETRAC) {
			/// UA-DETRAC
			// [int int float float float float float]
			// frame number
			// detection object id
			// x (top left) 
			// y (top left) 
			// width
			// height
			// confidence

			int frame_num_in_file = boost::lexical_cast<double>(vals.at(0));	// char * to int
			if (frame_num_in_file > iFrameNum) {
				//if (DEBUG_PRINT_MAIN) printf("%d!=%d\n", frame_num_in_file, iFrameNum);
				break;
			}

			bb[nObjects][0] = boost::lexical_cast<double>(vals.at(2)) /*- 1.0*/;	//if (bb[nObjects][0] < 0) bb[nObjects][0] = 0;
			bb[nObjects][1] = boost::lexical_cast<double>(vals.at(3)) /*- 1.0*/;	//if (bb[nObjects][1] < 0) bb[nObjects][1] = 0;
			bb[nObjects][2] = boost::lexical_cast<double>(vals.at(4));
			bb[nObjects][3] = boost::lexical_cast<double>(vals.at(5));
			bb[nObjects][4] = boost::lexical_cast<double>(vals.at(6));
			bb[nObjects][5] = OBJECT_TYPE_CAR;

			if (DEBUG_PRINT_MAIN)
				printf("[%d] %.lf %.lf %.lf %.lf %.2lf %.lf\n", frame_num_in_file, bb[nObjects][0], bb[nObjects][1], bb[nObjects][2], bb[nObjects][3], bb[nObjects][4], bb[nObjects][5]);

			if ((iter + 1) == allLines.end()) break;
		}
		nObjects++;
	}
	iter_last = iter;
	Nbbs = nObjects;

	//if (DEBUG_PRINT_MAIN) printf("# of Objects: %d\n", Nbbs);
	if (Nbbs > 0) bbs = (double*)malloc(sizeof(double)*Nbbs * 6);

	//if (DEBUG_PRINT_MAIN) printf("in -> ");
	for (int n = 0; n < Nbbs; n++) {
		// (x, y, width, height, confidence, object type)
		bbs[0 * Nbbs + n] = bb[n][0] / resize_factor;
		bbs[1 * Nbbs + n] = bb[n][1] / resize_factor;
		bbs[2 * Nbbs + n] = bb[n][2] / resize_factor;
		bbs[3 * Nbbs + n] = bb[n][3] / resize_factor;
		bbs[4 * Nbbs + n] = bb[n][4];
		bbs[5 * Nbbs + n] = bb[n][5];

		if (bb[n][5] == OBJECT_TYPE_PERSON)
			if (DEBUG_PRINT_MAIN)
				printf("[%d] %.lf %.lf %.lf %.lf\n", iFrameNum, bbs[0 * Nbbs + n], bbs[1 * Nbbs + n], bbs[2 * Nbbs + n], bbs[3 * Nbbs + n]);

	}
	//if (DEBUG_PRINT_MAIN) printf("out\n");
	free(tok);
	return bbs;
}
void WriteAllTracks2Txt(vector<vector<BBTrk>> allTracks, string DetName, string strThDetConf, string SeqName, int DB_TYPE) {
	if (DB_TYPE == DB_TYPE_MOT_CHALLENGE) {
		char filePath[256];
		sprintf_s(filePath, 256, "res\\MOT\\%s\\%s\\%s-%s.txt", DetName, strThDetConf, SeqName, DetName);

		cout << filePath << endl;

		FILE* fp = fopen(filePath, "w+");

		for (int i = 0; i < allTracks.size(); ++i) { // frame by frame

			if (!allTracks[i].empty()) {

				int tr = 0;
				for (int tr = 0; tr < allTracks[i].size(); ++tr) {

					fprintf(fp, "%d,%d,%.2lf,%.2lf,%.2lf,%.2lf,-1,-1,-1,-1\n", i + FRAME_OFFSET, allTracks[i][tr].id, (double)allTracks[i][tr].rec.x, (double)allTracks[i][tr].rec.y, (double)allTracks[i][tr].rec.width, (double)allTracks[i][tr].rec.height);
				}
			}
		}
		fclose(fp);

	}
	else if (DB_TYPE == DB_TYPE_UA_DETRAC) {
		int max_id = INT_MIN;
		for (int i = 0; i < allTracks.size(); ++i) { // frame by frame

			if (!allTracks[i].empty()) {

				// sort target by ID (ascending order)
				std::sort(allTracks[i].begin(), allTracks[i].end());


				if (max_id < allTracks[i].back().id) {
					max_id = allTracks[i].back().id;
				}

			}
			/*else {
			printf("[%d]",i);
			}*/
		}

		FILE* fp_LX = NULL, *fp_LY = NULL, *fp_W = NULL, *fp_H = NULL;

		char filePaths[4][256];
		sprintf_s(filePaths[0], 256, "res\\UA-DETRAC\\%s\\%s\\%s_LX.txt", DetName, strThDetConf, SeqName);
		sprintf_s(filePaths[1], 256, "res\\UA-DETRAC\\%s\\%s\\%s_LY.txt", DetName, strThDetConf, SeqName);
		sprintf_s(filePaths[2], 256, "res\\UA-DETRAC\\%s\\%s\\%s_W.txt", DetName, strThDetConf, SeqName);
		sprintf_s(filePaths[3], 256, "res\\UA-DETRAC\\%s\\%s\\%s_H.txt", DetName, strThDetConf, SeqName);

		cout << filePaths[0] << endl;
		cout << filePaths[1] << endl;
		cout << filePaths[2] << endl;
		cout << filePaths[3] << endl;

		fp_LX = fopen(filePaths[0], "w+");
		fp_LY = fopen(filePaths[1], "w+");
		fp_W = fopen(filePaths[2], "w+");
		fp_H = fopen(filePaths[3], "w+");

		for (int i = 0; i < allTracks.size(); ++i) { // frame by frame

			if (allTracks[i].empty()) {

				for (int tr = 0; tr < max_id; ++tr) {
					fprintf(fp_LX, "0,");
					fprintf(fp_LY, "0,");
					fprintf(fp_W, "0,");
					fprintf(fp_H, "0,");
				}
			}
			else {
				int tr = 0;
				for (int id = 0; id < max_id; ++id) {
					if (tr >= allTracks[i].size()) {
						fprintf(fp_LX, "0,");
						fprintf(fp_LY, "0,");
						fprintf(fp_W, "0,");
						fprintf(fp_H, "0,");
					}
					else {
						if (id == allTracks[i][tr].id) {
							fprintf(fp_LX, "%d,", (allTracks[i][tr].rec.x + 1));
							fprintf(fp_LY, "%d,", (allTracks[i][tr].rec.y + 1));
							fprintf(fp_W, "%d,", allTracks[i][tr].rec.width);
							fprintf(fp_H, "%d,", allTracks[i][tr].rec.height);
							tr++;
						}
						else {
							fprintf(fp_LX, "0,");
							fprintf(fp_LY, "0,");
							fprintf(fp_W, "0,");
							fprintf(fp_H, "0,");
						}
					}
				}
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
void ApproximateAllTracks(vector<vector<BBTrk>> inputTracks, vector<vector<BBTrk>>& outputTracks) {
	// Make vector<vector> tracks to map<vector> tracks
	map<int, vector<BBTrk>> allTracks;
	map<int, vector<BBTrk>> allTracksAppr;
	for (int i = 0; i < inputTracks.size(); ++i) {
		if (!inputTracks[i].empty()) {
			for (int j = 0; j < inputTracks[i].size(); ++j) {


				int id = inputTracks[i][j].id;

				vector<BBTrk> tracklet;
				tracklet.push_back(inputTracks[i][j]);

				pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = allTracks.insert(map<int, vector<BBTrk>>::value_type(id, tracklet));

				if (isEmpty.second == false) { // already has a element with target.at(j).id
					allTracks[id].push_back(inputTracks[i][j]);

				}
			}
		}
	}
	// Find the tracks' lost interval
	map<int, vector<BBTrk>>::iterator iterAllTrk;
	for (iterAllTrk = allTracks.begin(); iterAllTrk != allTracks.end(); ++iterAllTrk) {
		if (!iterAllTrk->second.empty()) {

			if (iterAllTrk->second.size() >= 2) {
				int i = 0;
				do {

					int prev_fn = iterAllTrk->second[i].fn;
					int cur_fn = iterAllTrk->second[i + 1].fn;

					if (prev_fn + 1 < cur_fn) {
						double fd = cur_fn - prev_fn;
						Rect prevRec = iterAllTrk->second[i].rec;
						Rect curRec = iterAllTrk->second[i + 1].rec;
						double xd = (curRec.x - prevRec.x) / fd;
						double yd = (curRec.y - prevRec.y) / fd;
						double wd = (curRec.width - prevRec.width) / fd;
						double hd = (curRec.height - prevRec.height) / fd;

						vector<BBTrk> trks[3]; // 0:track in pre-lost interval, 1: track in lost interval, 2: track in post-lost interval;
						int nSize = iterAllTrk->second.size();
						for (int f = 0; f < nSize; ++f) {
							if (f < i + 1)	trks[0].push_back(iterAllTrk->second[f]);
							else			trks[2].push_back(iterAllTrk->second[f]);
						}

						for (int f = 1; f < fd; ++f) {
							BBTrk appr = iterAllTrk->second[i];
							appr.fn = prev_fn + f;
							appr.rec.x = prevRec.x + f*xd;
							appr.rec.y = prevRec.y + f*yd;
							appr.rec.width = prevRec.width + f*wd;
							appr.rec.height = prevRec.height + f*hd;
							trks[1].push_back(appr);
						}
						vector<BBTrk> apprTrk;
						for (int t = 0; t < 3; t++) {
							for (int f = 0; f < trks[t].size(); f++) {
								apprTrk.push_back(trks[t][f]);
							}
						}
						iterAllTrk->second.clear();
						iterAllTrk->second.assign(apprTrk.begin(), apprTrk.end());
						i = i + fd - 1;
					}

					++i;
					if (i + 1 >= iterAllTrk->second.size()) break;
				} while (1);
			}

			vector<BBTrk> track = iterAllTrk->second;
			int id = iterAllTrk->first;
			pair< map<int, vector<BBTrk>>::iterator, bool> isEmpty = allTracksAppr.insert(map<int, vector<BBTrk>>::value_type(id, track));
		}
	}
	// Make map<vector> tracks to vector<vector> tracks
	map<int, vector<BBTrk>>::iterator iterMapAppr;
	for (int f = 0; f < inputTracks.size(); ++f) {
		map<int, vector<BBTrk>>::iterator iterMapAppr;
		vector<BBTrk> trkVecAtF;
		for (iterMapAppr = allTracksAppr.begin(); iterMapAppr != allTracksAppr.end(); ++iterMapAppr)
		{
			if (!iterMapAppr->second.empty()) {
				vector<BBTrk>::iterator iterT;
				for (iterT = iterMapAppr->second.begin(); iterT != iterMapAppr->second.end(); ++iterT) {
					if (iterT->fn == f) {
						trkVecAtF.push_back(*iterT);
					}
				}

			}
		}
		outputTracks.push_back(trkVecAtF);
	}
}
