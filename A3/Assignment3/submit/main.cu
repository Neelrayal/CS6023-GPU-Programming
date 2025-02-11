/* Workign single kernel code */

/*
	CS 6023 Assignment 3.
	Do not make any changes to the boiler plate code or the other files in the folder.
	Use cudaFree to deallocate any memory not in usage.
	Optimize as much as possible.
 */

#include "SceneNode.h"
#include <queue>
#include "Renderer.h"
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <chrono>

__global__ void fill_output_mat(int *opacitymat, int curopacity, int final_x, int final_y, int *mesh, int *gmat, int r, int c, int R, int C) {
    unsigned tid = blockIdx.x * 1024 + threadIdx.x;
    if ( tid >= r * c) return;

    int cur_row = tid / c + final_x;
    int cur_col = tid % c + final_y;

    if ( cur_row < 0 || cur_row >= R || cur_col < 0 || cur_col >= C)
      return;

    if ( opacitymat[cur_row* C + cur_col] <= curopacity){
      opacitymat[cur_row* C + cur_col] = curopacity;
      gmat[cur_row* C + cur_col] = mesh[ tid ];
    }
}

__global__ void solve(int *gmat, int R, int C){
  printf(" inside solve\n");
  for(int i = 0; i<R; ++i){
    for(int j = 0; j<C; ++j){
      printf("%d ", gmat[i*C + j]);
    }
    printf("\n");
  }
  printf("\n");
}


__global__ void pushChildren(int parent, int LIMIT, int *gupdates_r, int *gupdates_c, int *gOffset, int *gCsr, int *q, int *gcount){
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
		 if ( tid >= LIMIT ) return;
     int curChild =  gCsr [ gOffset[parent] + tid ];
		 gupdates_r[curChild] += gupdates_r[parent];
		 gupdates_c[curChild] += gupdates_c[parent];
     int qIndex = atomicAdd(gcount, 1);
     q[qIndex] = curChild;
}

__global__ void traverse(int ptr, int LIMIT, int *gupdates_r, int *gupdates_c, int *gOffset, int *gCsr, int *q, int *gcount ){
		 unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
		 if ( tid >= LIMIT ) return;

		 int parent =  q[ptr + tid];

		 int children = gOffset[parent+1] - gOffset[parent];
		 if ( children == 0) return;

			/*
		 int blocks = ceil ( (children / (float) 1024) );
		 pushChildren <<< blocks, 1024 >>> (parent, children , gupdates_r, gupdates_c, gOffset, gCsr, q, gcount);
		 */
		for(int i = 0; i<children; ++i){
				 int tid = i;
				 int curChild =  gCsr [ gOffset[parent] + tid ];
					gupdates_r[curChild] += gupdates_r[parent];
					gupdates_c[curChild] += gupdates_c[parent];
					int qIndex = atomicAdd(gcount, 1);
					q[qIndex] = curChild;
		}
}

__global__ void initialize (int *gcount, int *q){
    *gcount = 1;
    q[0] = 0;
}


void readFile (const char *fileName, std::vector<SceneNode*> &scenes, std::vector<std::vector<int> > &edges, std::vector<std::vector<int> > &translations, int &frameSizeX, int &frameSizeY) {
	/* Function for parsing input file*/

	FILE *inputFile = NULL;
	// Read the file for input.
	if ((inputFile = fopen (fileName, "r")) == NULL) {
		printf ("Failed at opening the file %s\n", fileName) ;
		return ;
	}

	// Input the header information.
	int numMeshes ;
	fscanf (inputFile, "%d", &numMeshes) ;
	fscanf (inputFile, "%d %d", &frameSizeX, &frameSizeY) ; // R * C


	// Input all meshes and store them inside a vector.
	int meshX, meshY ;
	int globalPositionX, globalPositionY; // top left corner of the matrix.
	int opacity ;
	int* currMesh ;
	for (int i=0; i<numMeshes; i++) {
		fscanf (inputFile, "%d %d", &meshX, &meshY) ; // r, c size of each mesh
		fscanf (inputFile, "%d %d", &globalPositionX, &globalPositionY) ; // x, y starting co-ordinate of each mesh
		fscanf (inputFile, "%d", &opacity) ;
		currMesh = (int*) malloc (sizeof (int) * meshX * meshY) ; // 2-d matrix mesh
		for (int j=0; j<meshX; j++) {
			for (int k=0; k<meshY; k++) {
				fscanf (inputFile, "%d", &currMesh[j*meshY+k]) ;
        //printf("%d ", currMesh[j*meshY+k]);
			}
      //printf("\n");
		}
		//Create a Scene out of the mesh.
		SceneNode* scene = new SceneNode (i, currMesh, meshX, meshY, globalPositionX, globalPositionY, opacity) ;
		scenes.push_back (scene) ; // array of meshes
	}

	// Input all relations and store them in edges.
	int relations;
	fscanf (inputFile, "%d", &relations) ; // E which is #edges
	int u, v ;
	for (int i=0; i<relations; i++) {
		fscanf (inputFile, "%d %d", &u, &v) ;
		edges.push_back ({u,v}) ;
	}

	// Input all translations.
	int numTranslations ;
	fscanf (inputFile, "%d", &numTranslations) ;
	std::vector<int> command (3, 0) ;
	for (int i=0; i<numTranslations; i++) {
		fscanf (inputFile, "%d %d %d", &command[0], &command[1], &command[2]) ;
		translations.push_back (command) ;
	}
}


void writeFile (const char* outputFileName, int *hFinalPng, int frameSizeX, int frameSizeY) {
	/* Function for writing the final png into a file.*/
	FILE *outputFile = NULL;
	if ((outputFile = fopen (outputFileName, "w")) == NULL) {
		printf ("Failed while opening output file\n") ;
	}

	for (int i=0; i<frameSizeX; i++) {
		for (int j=0; j<frameSizeY; j++) {
			fprintf (outputFile, "%d ", hFinalPng[i*frameSizeY+j]) ;
		}
		fprintf (outputFile, "\n") ;
	}
}


int main (int argc, char **argv) {

	// Read the scenes into memory from File.
	const char *inputFileName = argv[1] ;
	int* hFinalPng;

	int frameSizeX, frameSizeY ; // R * C
	std::vector<SceneNode*> scenes ;
	std::vector<std::vector<int> > edges ;
	std::vector<std::vector<int> > translations ;
	readFile (inputFileName, scenes, edges, translations, frameSizeX, frameSizeY) ;
	hFinalPng = (int*) malloc (sizeof (int) * frameSizeX * frameSizeY) ;

	// Make the scene graph from the matrices.
    Renderer* scene = new Renderer(scenes, edges) ;

	// Basic information.
	int V = scenes.size () ; // #meshes
	int E = edges.size () ; // #edges
	int numTranslations = translations.size () ; // #queries

	// Convert the scene graph into a csr.
	scene->make_csr () ; // Returns the Compressed Sparse Row representation for the graph.
	int *hOffset = scene->get_h_offset () ;
	int *hCsr = scene->get_h_csr () ;
	int *hOpacity = scene->get_opacity () ; // hOpacity[vertexNumber] contains opacity of vertex vertexNumber.
	int **hMesh = scene->get_mesh_csr () ; // hMesh[vertexNumber] contains the mesh attached to vertex vertexNumber.
	int *hGlobalCoordinatesX = scene->getGlobalCoordinatesX () ; // hGlobalCoordinatesX[vertexNumber] contains the X coordinate of the vertex vertexNumber.
	int *hGlobalCoordinatesY = scene->getGlobalCoordinatesY () ; // hGlobalCoordinatesY[vertexNumber] contains the Y coordinate of the vertex vertexNumber.
	int *hFrameSizeX = scene->getFrameSizeX () ; // hFrameSizeX[vertexNumber] contains the vertical size of the mesh attached to vertex vertexNumber.
	int *hFrameSizeY = scene->getFrameSizeY () ; // hFrameSizeY[vertexNumber] contains the horizontal size of the mesh attached to vertex vertexNumber.

	auto start = std::chrono::high_resolution_clock::now () ;


	// Code begins here.
	// Do not change anything above this comment.

  // giving memory to our final matrix on GPU
  
  int *updates_r = (int *) malloc(V * sizeof(int));
  int *updates_c = (int *) malloc(V * sizeof(int));

  for(int i = 0; i<V; ++i){
    updates_c[i] = 0;
    updates_r[i] = 0;
  }

  for(int i = 0; i<numTranslations; ++i){
      int meshid = translations[i][0], command = translations[i][1], amount = translations[i][2];
      switch(command){
          case 0:
            updates_r[meshid] -= amount;
            break;
          case 1:
            updates_r[meshid] += amount;
            break;
          case 2:
            updates_c[meshid] -= amount;
            break;
          case 3:
            updates_c[meshid] += amount;
            break;
          default:
            printf(" %d problem\n", command);
            break;
      }
  }

	int *gupdates_r, *gupdates_c;
	cudaMalloc(&gupdates_r, V * sizeof(int));
	cudaMalloc(&gupdates_c, V * sizeof(int));
	cudaMemcpy(gupdates_r, updates_r, V * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gupdates_c, updates_c, V * sizeof(int), cudaMemcpyHostToDevice);
  
	int *gCsr, *gOffset;
	cudaMalloc(&gCsr, E * sizeof(int));
	cudaMalloc(&gOffset, (V + 1) * sizeof(int));
	cudaMemcpy(gCsr, hCsr, E * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gOffset, hOffset, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);

	int *gcount, *q;
	cudaMalloc(&gcount, sizeof(int));
 	cudaMalloc(&q, (V + 5 ) * sizeof(int));
	initialize <<< 1 , 1 >>> (gcount, q);
	cudaDeviceSynchronize();

	int ptr = 0, diff = 1, old_count, new_count;
	while ( diff ){
		cudaMemcpy(&old_count, gcount, sizeof(int), cudaMemcpyDeviceToHost);
	  int blocks = ceil ( (diff / (float) 1024) );
    traverse<<< blocks, 1024 >>> (ptr, diff, gupdates_r, gupdates_c, gOffset, gCsr, q, gcount);
		cudaDeviceSynchronize();
    cudaMemcpy(&new_count, gcount, sizeof(int), cudaMemcpyDeviceToHost);
		diff = new_count - old_count;
		ptr = old_count;
	}

	/*
	int children = hOffset[1]-hOffset[0];
	int blocks = ceil ( (children / (float) 1024) );

	traverse <<< blocks, 1024 >>> (0, children , gupdates_r, gupdates_c, gOffset, gCsr, updates_r[0], updates_c[0]);
	*/
	cudaMemcpy(updates_r, gupdates_r, V * sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(updates_c, gupdates_c, V * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(q);
  cudaFree(gCsr);
  cudaFree(gOffset);
  cudaFree(gupdates_r);
  cudaFree(gupdates_c);
	/*
	for(int i = 0; i<V; ++i){
			 printf("%d %d\n", updates_r[i], updates_c[i]);
	}
	*/

	int *opacitymat;
  cudaMalloc(&opacitymat, frameSizeX * frameSizeY* sizeof(int));
	cudaMemset(opacitymat, 0, frameSizeX * frameSizeY* sizeof(int));

	int *gmat;
  cudaMalloc(&gmat, frameSizeX * frameSizeY* sizeof(int));
	cudaMemset(gmat, 0, frameSizeX * frameSizeY* sizeof(int));


  //printf("value of V: %d\n", V);
	int *gtempmesh; 
	cudaMalloc(&gtempmesh, (10005) * sizeof(int));
  for(int i = 0; i<V; ++i){
    int curmeshId = i;
    int r = hFrameSizeX[curmeshId];
    int c = hFrameSizeY[curmeshId];		
		cudaMalloc(&gtempmesh, r * c * sizeof(int));
		cudaMemcpy(gtempmesh, hMesh[curmeshId], r * c * sizeof(int), cudaMemcpyHostToDevice);
    int blocks = ceil (r*c/(float)1024);
    int final_x = updates_r[i] + hGlobalCoordinatesX[i];
    int final_y = updates_c[i] + hGlobalCoordinatesY[i];

    //printf(" Mesh = %d\n", i);
    fill_output_mat<<< blocks, 1024 >>> (opacitymat, hOpacity[curmeshId], final_x, final_y, gtempmesh, gmat,
        r, c, frameSizeX, frameSizeY);
      cudaDeviceSynchronize();
    //solve <<< 1 , 1 >>> (gmat, frameSizeX, frameSizeY);

  }

  cudaMemcpy(hFinalPng, gmat, frameSizeX*frameSizeY*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(opacitymat);
  cudaFree(gmat);

	free(updates_c);
	free(updates_r);

	// Do not change anything below this comment.
	// Code ends here.

	auto end  = std::chrono::high_resolution_clock::now () ;

	std::chrono::duration<double, std::micro> timeTaken = end-start;

	printf ("execution time : %f\n", timeTaken) ;
	// Write output matrix to file.
	const char *outputFileName = argv[2] ;
	writeFile (outputFileName, hFinalPng, frameSizeX, frameSizeY) ;

}
