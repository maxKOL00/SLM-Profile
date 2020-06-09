#include "cudaCore.cuh"
#include "Thrower.h"
#include <fstream>
#include <math.h>
#include <random>
#include <complex>



cufftDoubleComplex evaluate_complex_exponetial(double phase, double amplitude) {
    double real = amplitude * cos(phase);
    double imag = amplitude * sin(phase);
    cufftDoubleComplex value;
    value.x = real;
    value.y = imag;
    return value;
}

#define IDX2R(i,j,N) (((i)*(N))+(j))
__global__ void fftshift_2D(cufftDoubleComplex* data, int N)
{
    int i = threadIdx.y + blockDim.y * blockIdx.y;
    int j = threadIdx.x + blockDim.x * blockIdx.x;

    if (i < N && j < N)
    {
        double a = 1 - 2 * ((i + j) & 1);
        //double a = pow(-1.0, (i + j) & 1); above is faster
        data[IDX2R(i, j, N)].x *= a;
        data[IDX2R(i, j, N)].y *= a;
    }
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

double random_phase() {
    double lower_bound = 0;
    double upper_bound = 3.14159;
    std::random_device rd;  //Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> unif(lower_bound, upper_bound);
    double a_random_double = unif(gen) - unif(gen);
    return a_random_double;
}

void cudaCore::createArrays() {
    //calculate the number of pixels between the traps in target image
    //resolution = 108695*850nm*20mm/(telescope_mag*active)
    double resolution = (108695.0 * 850 * pow(10, -9) * 0.020) / (3.3 * NX);//not sure if padded or unpadded size
    int num_pix = trap_spacing * 0.000001 / resolution;
    if (num_pix < 1) {
        num_pix = 2;
    }
    if (dim % 2 == 0) {
        st = int(mid - ((dim / 2) * num_pix) + (num_pix/2));
    }
    else {
        st = int(mid - ((dim / 2) * num_pix));//find the upper left corner of the trap array
    }
    if (st < 0) {
        thrower("the number of traps is too large for the given resolution and trap spacing");
    }
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            double gauss = gaussian(y_coord, x_coord);
            input_intensity[y_coord * NX + x_coord] = gauss;
            cufftDoubleComplex value = evaluate_complex_exponetial(random_phase(), gauss);
            DiffPlane[y_coord * NX + x_coord].x = value.x;
            DiffPlane[y_coord * NX + x_coord].y = value.y;
            if (x_coord >= st && y_coord >= st && x_coord <= (st + ((dim - 1) * num_pix)) && y_coord <= (st + ((dim - 1) * num_pix))) {
                if ((((x_coord - st) % int(num_pix)) == 0)) {
                    if (((y_coord - st) % int(num_pix)) == 0) {
                        target[y_coord * NX + x_coord] = 1;
                        st = st;
                    }
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
    mid = nx / 2;
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
        if (GPU_name == "GeForce RTX 2080 SUPER") {
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
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);
    cufftDoubleComplex* diffTemp;
    cudaMalloc((void**)&diffTemp, sizeof(cufftDoubleComplex) * NX * (NY));
    err = cudaMemcpy(diffTemp, DiffPlane, sizeof(cufftDoubleComplex) * NX * (NY), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        thrower("Cuda error: Failed to allocate diffPlane data\n");
    }
    fftshift_2D << <numBlocks, threadsPerBlock >> > (diffTemp, NX);//compiles fine
    cufftExecZ2Z(plan, diffTemp, ImagePlane, CUFFT_FORWARD);
    fftshift_2D << <numBlocks, threadsPerBlock >> > (ImagePlane, NX);
    cudaFree(diffTemp);
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
    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);
    cufftDoubleComplex* ImgTemp;
    cudaMalloc((void**)&ImgTemp, sizeof(cufftDoubleComplex) * NX * (NY));
    err = cudaMemcpy(ImgTemp, ImagePlane, sizeof(cufftDoubleComplex) * NX * (NY), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        thrower("Cuda error: Failed to allocate diffPlane data\n");
    }
    fftshift_2D << <numBlocks, threadsPerBlock >> > (ImgTemp, NX);//compiles fine
    cufftExecZ2Z(plan, ImgTemp, DiffPlane, CUFFT_INVERSE);
    fftshift_2D << <numBlocks, threadsPerBlock >> > (DiffPlane, NX);
    cudaFree(ImgTemp);

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

__global__ void newPhase(cufftDoubleReal* input_intensity, cufftDoubleComplex* DiffPlane, int N) {
    //IDK if this is the fastest way
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    cufftDoubleComplex value;
    double phase;
    if (i < N && j < N) {
        phase = atan2(DiffPlane[i * N + j].y, DiffPlane[i * N + j].x);
        value.x = input_intensity[i * N + j] * cos(phase);
        value.y = input_intensity[i * N + j] * sin(phase);
        DiffPlane[i * N + j] = value;
    }
}


__global__ void newAmp(cufftDoubleReal* target, cufftDoubleComplex* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    cufftDoubleComplex value;
    double phase;
    if (i < N && j < N) {
        phase = atan2(C[i * N + j].y , C[i * N + j].x);
        value.x = target[i * N + j] * cos(phase);
        value.y = target[i * N + j] * sin(phase);
        C[i * N + j] = value;
    }
}

void cudaCore::swapPhase() {

    dim3 threadsPerBlock(16, 16);
    size_t free, total;
    dim3 numBlocks(NX / threadsPerBlock.x, NY / threadsPerBlock.y);
    cudaMemGetInfo(&free, &total);
    
    newPhase << <numBlocks, threadsPerBlock >> > (input_intensity, DiffPlane, NX);//compiles fine
    if (cudaThreadSynchronize() != cudaSuccess) {//Otherwise the cpu can't find where the data went
        thrower("Cuda error: Failed to synchronize\n");
    }
    cudaMemGetInfo(&free, &total);
    
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
    int num_pix = trap_spacing * 0.000001 / resolution;

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
    mid = size / 2;
}
void cudaCore::setActiveSize(int size) {
    activeN = size;
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
    int num_pix = trap_spacing * 0.000001 / resolution;
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
    cudaDeviceSynchronize();
    std::ofstream file("DiffPlaneValues.txt", std::ofstream::out | std::ofstream::trunc);
    double real, imag;
    double value;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            real = DiffPlane[y_coord * NX + x_coord].x;
            imag = DiffPlane[y_coord * NX + x_coord].y;
            value = (pow(real, 2) + pow(imag, 2));
            file << value << " ";
        }
        file << std::endl;
    }
    file.close();
}

void cudaCore::saveImageBitmap() {
    //cudaDeviceSynchronize();
    std::ofstream file("ImagePlaneValues.txt", std::ofstream::out | std::ofstream::trunc);
    double real, imag;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            real = ImagePlane[y_coord * NX + x_coord].x;
            imag = ImagePlane[y_coord * NX + x_coord].y;
            file << (real*real) + (imag * imag) << " ";
        }
        file << std::endl;
    }
    file.close();
}

void cudaCore::saveTarget() {
    //cudaDeviceSynchronize();
    std::ofstream file("Target.txt", std::ofstream::out | std::ofstream::trunc);
    double real;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            real = target[y_coord * NX + x_coord];
            file << real << " ";
        }
        file << std::endl;
    }
    file.close();
}

void cudaCore::savePhasePlot() {
    //cudaDeviceSynchronize();
    std::ofstream file("Phase.txt", std::ofstream::out | std::ofstream::trunc);
    double phase;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            phase = atan2(DiffPlane[y_coord * NX + x_coord].y, DiffPlane[y_coord * NX + x_coord].x);
            file << phase << " ";
        }
        file << std::endl;
    }
    file.close();
}

void cudaCore::savePhaseN() {
    double phase;
    for (size_t y_coord = 0; y_coord < NY; y_coord++) {
        for (size_t x_coord = 0; x_coord < NX; x_coord++) {
            phase = atan2(ImagePlane[y_coord * NX + x_coord].y, ImagePlane[y_coord * NX + x_coord].x);
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