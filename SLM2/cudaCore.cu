#include "cudaCore.cuh"
#include "Thrower.h"
#include <fstream>
#include <math.h>



cufftDoubleComplex evaluate_complex_exponetial(double phase, double amplitude) {
    double real = amplitude * cos(phase);
    double imag = amplitude * sin(phase);
    cufftDoubleComplex value;
    value.x = real;
    value.y = imag;
    return value;
}

double cudaCore::gaussian(int i, int j) {
    //x0 = active / 2
    //y0 = active / 2 #center coordinates
    //sigma = active / 2 #beam waist
    //A = 1 #is the peak aplitude
    //since the gaussian is centered in the active area and 
    //has 1/e^2 waist equal to active area all these variables equal mid
    /*res = (((i - x0) * *2) + ((j - y0) * *2)) / (2 * (sigma * *2))
     value = A * np.exp(-1 * res)*/
    if (pow(i - mid, 2) + pow(j - mid, 2) > pow(mid, 2)) {
        return 0;
    }
    double res = (pow(i - mid, 2) + pow(j - mid, 2)) / (2 * pow(mid, 2));
    return exp(-1 * res);
}

void cudaCore::createArrays() {
    //calculate the number of pixels between the traps in target image
    //resolution = 108695*850nm*20mm/(telescope_mag*active)
    double resolution = (108695.0 * 850 * pow(10, -9) * 0.020) / (3.3 * NX);//not sure if padded or unpadded size
    int num_pix = trap_spacing / resolution;
    if (num_pix < 1) {
        num_pix = 2;
    }
    int st = int(mid - ((dim / 2) * num_pix));//find the upper left corner of the trap array
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            double gauss = gaussian(y_coord, x_coord);
            input_intensity[y_coord * NX + x_coord] = gauss;
            cufftDoubleComplex value = evaluate_complex_exponetial(rand() % 6, gauss);
            DiffPlane[y_coord * NX + x_coord].x = value.x;
            DiffPlane[y_coord * NX + x_coord].y = value.y;
            if (x_coord >= st && y_coord >= st && x_coord <= (st + ((dim - 1) * num_pix)) && y_coord <= (st + ((dim - 1) * num_pix))) {
                if ((((x_coord - st) % dim) == 0) && (((y_coord - st) % dim) == 0)) {
                    target[y_coord * NX + x_coord] == 1;
                }
            }
            else {
                target[y_coord * NX + x_coord] == 0;
            }
        }
    }
}

cudaCore::cudaCore(size_t nx, size_t ny, int activen, double spacing, int Dim) {
    NX = nx;
    NY = ny;
    trap_spacing = spacing;
    dim = Dim;
    /*const size_t row_pointers_bytes = NY * sizeof * phaseN;
    const size_t row_elements_bytes = NX * sizeof * *phaseN;*/
    phaseN = new double[NY * NX];
    if (Dim < 0) {dim = 10;}
    int count = 0;
    cudaGetDeviceCount(&count);//expect two GPUs 
    cudaDeviceProp prop;
    for (int i = 0; i < count; i++) {
        err = cudaGetDeviceProperties(&prop, i);
        if (err != cudaSuccess) {
            thrower("CUDA Error on %s\n", cudaGetErrorString(err));
        }
        GPU_name = (prop.name);
        if (GPU_name == "NVIDIA geforce 2080") {
            cudaSetDevice(i);
            break;
        }
    }
    //mallocManaged call allocated data so that both gpu and cpu can have accesss
    err = cudaMallocManaged(&DiffPlane, sizeof(cufftDoubleComplex) * NX * (NY));
    if (err != cudaSuccess) {
        thrower("CUDA Error on %s\n", cudaGetErrorString(err));
    }
    err = cudaMallocManaged(&ImagePlane, sizeof(cufftDoubleComplex) * NX * (NY));
    if (err != cudaSuccess) {
        thrower("Cuda error: Failed to allocate ImagePlane data\n");
    }
    err = cudaMallocManaged(&input_intensity, sizeof(cufftDoubleReal) * NX * (NY));
    if (err != cudaSuccess) {
        thrower("Cuda error: Failed to allocate ImagePlane data\n");
    }
    err = cudaMallocManaged(&target, sizeof(cufftDoubleReal) * NX * (NY));
    if (err != cudaSuccess) {
        thrower("Cuda error: Failed to allocate ImagePlane data\n");
    }
    cudaDeviceSynchronize();//make sure the cpu and gpu know where to find the data
    cufftPlan2d(&plan, NX, NY, CUFFT_Z2Z);//Done in advance to tell the fft how to execute

    ///initialize the diffraction array and the target image
    //calculate the number of pixels between the traps in target image
    //resolution = 108695*850nm*20mm/(telescope_mag*active)
    createArrays();

}

cudaCore::~cudaCore() {
    //free the data
    cufftDestroy(plan);
    cudaFree(DiffPlane);
    cudaFree(ImagePlane);
    delete[] phaseN;
}

cufftDoubleComplex* cudaCore::get_image_plane() {
    return ImagePlane;
}
cufftDoubleComplex* cudaCore::get_diff_plane() {
    return DiffPlane;
}

void cudaCore::execfft() {
    cufftExecZ2Z(plan, DiffPlane, ImagePlane, CUFFT_FORWARD);
    if (cudaGetLastError() != cudaSuccess) {
        thrower("Cuda error: Failed to execute fft\n");
    }
    if (cudaThreadSynchronize() != cudaSuccess) {//Otherwise the cpu can't find where the data went
        thrower("Cuda error: Failed to synchronize\n");
    }
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t i = 0; i < NX; i++) {
            ImagePlane[y_coord * NX + i].x = ImagePlane[y_coord * NX + i].x / norm;
            ImagePlane[y_coord * NX + i].y = ImagePlane[y_coord * NX + i].y / norm;
        }
    }
}

void cudaCore::execifft() {
    cufftExecZ2Z(plan, ImagePlane, DiffPlane, CUFFT_INVERSE);
    if (cudaGetLastError() != cudaSuccess) {
        thrower("Cuda error: Failed to execute fft\n");
    }
    if (cudaThreadSynchronize() != cudaSuccess) {//Otherwise the cpu can't find where the data went
        thrower("Cuda error: Failed to synchronize\n");
    }
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t i = 0; i < NX; i++) {
            DiffPlane[y_coord * NX + i].x = DiffPlane[y_coord * NX + i].x / norm;
            DiffPlane[y_coord * NX + i].y = DiffPlane[y_coord * NX + i].y / norm;
        }
    }
}

__global__ void newPhase(cufftDoubleReal* amp, cufftDoubleComplex* B, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    cufftDoubleComplex value;
    double phase;
    if (i < N && j < N) {
        phase = atan(B[i * N + j].y / B[i * N + j].x);
        value.x = amp[i * N + j] * cos(phase);
        value.y = amp[i * N + j] * sin(phase);
        B[i * N + j] = value;
    }
}

void cudaCore::swapPhase() {

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);

    newPhase << <numBlocks, threadsPerBlock >> > (input_intensity, DiffPlane, NX);//compiles fine
    if (cudaThreadSynchronize() != cudaSuccess) {//Otherwise the cpu can't find where the data went
        thrower("Cuda error: Failed to synchronize\n");
    }
}

__global__ void newAmp(cufftDoubleReal* target, cufftDoubleComplex* C, int N) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    cufftDoubleComplex value;
    double phase;
    if (i < N && j < N) {
        phase = atan(C[i * N + j].y / C[i * N + j].x);
        value.x = target[i * N + j] * cos(phase);
        value.y = target[i * N + j] * sin(phase);
        C[i * N + j] = value;
    }
}

void cudaCore::swapAmp() {
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);

    newAmp << <numBlocks, threadsPerBlock >> > (target, ImagePlane, NX);//apparently just intellisense error
    if (cudaThreadSynchronize() != cudaSuccess) {//Otherwise the cpu can't find where the data went
        thrower("Cuda error: Failed to synchronize\n");
    }
}


void cudaCore::weightImage() {
    double resolution = (108695.0 * 850 * pow(10, -9) * 0.020) / (3.3 * NX);//not sure if padded or unpadded size
    int num_pix = trap_spacing / resolution;
    int st = int(mid - ((dim / 2) * num_pix));//find the upper left corner of the trap array
    double total = 0;
    for (size_t y_coord = 0; y_coord < dim; y_coord++) {
        for (size_t x_coord = 0; x_coord < dim; x_coord++) {
            total = total + pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].x, 2)
                + pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].y, 2);
        }
    }
    double I_bar = total / (dim * dim);
    double denom;
    double Ii;
    for (size_t y_coord = 0; y_coord < dim; y_coord++) {
        for (size_t x_coord = 0; x_coord < dim; x_coord++) {
            Ii = pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].x, 2)
                + pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].y, 2);
            denom = (1 - (0.7 * (1 - (Ii / I_bar))));
            target[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)] =
                target[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)] * (I_bar / denom);
        }
    }
}

void cudaCore::setPaddedSize(int size) {
    NX = size;
    NY = size;
    norm = NX * NY;
}
void cudaCore::setActiveSize(int size) {
    activeN = size;
    mid = activeN / 2;
}
void cudaCore::setTrapSpacing(double spacing) {
    trap_spacing = spacing;
}
void cudaCore::setDim(int Dim) {
    dim = Dim;
}

void cudaCore::weightImage2() {
    //Englund just used the average amplitude over the amp. of the trap.
    double resolution = (108695.0 * 850 * pow(10, -9) * 0.020) / (3.3 * NX);//not sure if padded or unpadded size
    int num_pix = trap_spacing / resolution;
    int st = int(mid - ((dim / 2) * num_pix));//find the upper left corner of the trap array
    double total = 0;
    for (size_t y_coord = 0; y_coord < dim; y_coord++) {
        for (size_t x_coord = 0; x_coord < dim; x_coord++) {

            total = total + sqrt(pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].x, 2)
                                 + pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].y, 2));
        }
    }
    double Amp_bar = total / (dim * dim);
    double weight;
    double Ai;
    for (size_t y_coord = 0; y_coord < dim; y_coord++) {
        for (size_t x_coord = 0; x_coord < dim; x_coord++) {

            Ai = sqrt(pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].x, 2)
                      + pow(ImagePlane[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)].y, 2));
            weight = (Ai / Amp_bar);
            target[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)] =
                weight * target[((st + num_pix * y_coord) * NX) + (st + num_pix * x_coord)];
        }
    }
}

void cudaCore::saveDiffPlane() {
    std::ofstream file("DiffPlaneValues.txt");
    double real, imag;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            real = DiffPlane[y_coord * NX + x_coord].x;
            imag = DiffPlane[y_coord * NX + x_coord].y;
            file << (pow(real, 2) + pow(imag, 2)) << "/n";
        }
    }
    file.close();
}

void cudaCore::saveImageBitmap() {
    std::ofstream file("ImagePlaneValues.txt");
    double real, imag;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            real = ImagePlane[y_coord * NX + x_coord].x;
            imag = ImagePlane[y_coord * NX + x_coord].y;
            file << (pow(real, 2) + pow(imag, 2)) << "/n";
        }
    }
    file.close();
}

void cudaCore::savePhaseN() {
    double phase;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            phase = atan(ImagePlane[y_coord * NX + x_coord].y / ImagePlane[y_coord * NX + x_coord].x);
            *(phaseN + y_coord * NX + x_coord) = phase;
        }
    }
}

void cudaCore::swapPhaseN() {
    cufftDoubleComplex value;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            value.x = target[y_coord * NX + x_coord] * cos(*(phaseN + y_coord * NX + x_coord));
            value.y = target[y_coord * NX + x_coord] * sin(*(phaseN + y_coord * NX + x_coord));
            ImagePlane[y_coord * NX + x_coord] = value;
        }
    }
}