//
//  ViewController.m
//  MNIST
//
//  Created by Matt on 11/24/16.
//  Copyright © 2016 Matt Rajca. All rights reserved.
//

#import "ViewController.h"

#include <tensorflow/core/public/session.h>
#include <tensorflow/core/platform/env.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <Accelerate/Accelerate.h>

using namespace tensorflow;

static constexpr int kUsedExamples = 5000;
static constexpr int kImageSide = 28;
static constexpr int kImageSide2 = kImageSide / 2;
static constexpr int kImageSide4 = kImageSide / 4;
static constexpr int kOutputs = 10;
static constexpr int kInputLength = kImageSide * kImageSide;

@implementation ViewController

- (IBAction)test:(id)sender {
	Session* session;
	Status status = NewSession(SessionOptions(), &session);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return;
	}

	NSString *modelPath = [[NSBundle mainBundle] pathForResource:@"final" ofType:@"pb"];

	GraphDef graph;
	status = ReadBinaryProto(Env::Default(), modelPath.fileSystemRepresentation, &graph);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return;
	}

	status = session->Create(graph);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return;
	}

	Tensor x(DT_FLOAT, TensorShape({ kUsedExamples, kInputLength }));

	NSString *imagesPath = [[NSBundle mainBundle] pathForResource:@"images" ofType:nil];
	NSString *labelsPath = [[NSBundle mainBundle] pathForResource:@"labels" ofType:nil];
	NSData *imageData = [NSData dataWithContentsOfFile:imagesPath];
	NSData *labelsData = [NSData dataWithContentsOfFile:labelsPath];

	uint8_t *expectedLabels = new uint8_t[kUsedExamples];

	for (auto exampleIndex = 0; exampleIndex < kUsedExamples; exampleIndex++) {
		// Actual labels start at offset 8.
		[labelsData getBytes:&expectedLabels[exampleIndex] range:NSMakeRange(8 + exampleIndex, 1)];

		for (auto i = 0; i < kInputLength; i++) {
			uint8_t pixel;
			// Actual image data starts at offset 16.
			[imageData getBytes:&pixel range:NSMakeRange(16 + exampleIndex * kInputLength + i, 1)];
			x.matrix<float>().operator()(exampleIndex, i) = pixel / 255.0f;
		}
	}

	std::vector<std::pair<string, Tensor>> inputs = {
		{ "x", x }
	};

	const auto start = CACurrentMediaTime();

	std::vector<Tensor> outputs;
	status = session->Run(inputs, {"softmax"}, {}, &outputs);
	if (!status.ok()) {
		std::cout << status.ToString() << "\n";
		return;
	}

	NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);

	const auto outputMatrix = outputs[0].matrix<float>();
	int correctExamples = 0;

	for (auto exampleIndex = 0; exampleIndex < kUsedExamples; exampleIndex++) {
		int bestIndex = -1;
		float bestProbability = 0;
		for (auto i = 0; i < kOutputs; i++) {
			const auto probability = outputMatrix(exampleIndex, i);
			if (probability > bestProbability) {
				bestProbability = probability;
				bestIndex = i;
			}
		}

		if (bestIndex == expectedLabels[exampleIndex]) {
			correctExamples++;
		}
	}

	NSLog(@"Accuracy: %f", static_cast<float>(correctExamples) / kUsedExamples);

	delete[] expectedLabels;
	session->Close();
}

static float *loadTensor(NSString *baseName, NSUInteger length) {
	NSString *path = [[NSBundle mainBundle] pathForResource:baseName ofType:nil];
	NSData *data = [NSData dataWithContentsOfFile:path];
	float *tensor = new float[length];
	for (NSUInteger i = 0; i < length; i++) {
		[data getBytes:&tensor[i] range:NSMakeRange(i * sizeof(float), sizeof(float))];
	}

	return tensor;
}

- (IBAction)testGPU:(id)sender {
	float *conv1weights = loadTensor(@"W_conv1", 5 * 5 * 1 * 32);
	float *conv1biases = loadTensor(@"b_conv1", 32);
	float *conv2weights = loadTensor(@"W_conv2", 5 * 5 * 32 * 64);
	float *conv2biases = loadTensor(@"b_conv2", 64);
	float *fc1weights = loadTensor(@"W_fc1", 7 * 7 * 64 * 1024);
	float *fc1biases = loadTensor(@"b_fc1", 1024);
	float *fc2weights = loadTensor(@"W_fc2", 1024 * 10);
	float *fc2biases = loadTensor(@"b_fc2", 10);

	NSString *imagesPath = [[NSBundle mainBundle] pathForResource:@"images" ofType:nil];
	NSString *labelsPath = [[NSBundle mainBundle] pathForResource:@"labels" ofType:nil];
	NSData *imageData = [NSData dataWithContentsOfFile:imagesPath];
	NSData *labelsData = [NSData dataWithContentsOfFile:labelsPath];

	uint8_t *expectedLabels = new uint8_t[kUsedExamples];

	float *x = new float[kUsedExamples * kInputLength];
	size_t xIndex = 0;

	for (auto exampleIndex = 0; exampleIndex < kUsedExamples; exampleIndex++) {
		// Actual labels start at offset 8.
		[labelsData getBytes:&expectedLabels[exampleIndex] range:NSMakeRange(8 + exampleIndex, 1)];

		for (auto i = 0; i < kInputLength; i++) {
			uint8_t pixel;
			// Actual image data starts at offset 16.
			[imageData getBytes:&pixel range:NSMakeRange(16 + exampleIndex * kInputLength + i, 1)];
			x[xIndex++] = pixel / 255.0f;
		}
	}

	id<MTLDevice> device = MTLCreateSystemDefaultDevice();
	if (device == nil) {
		NSLog(@"No Metal device");
		return;
	}

	id<MTLCommandQueue> queue = [device newCommandQueue];

	const MPSCNNNeuronReLU *reluUnit = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:0];

	MPSCNNConvolutionDescriptor *conv1descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:5 kernelHeight:5 inputFeatureChannels:1 outputFeatureChannels:32 neuronFilter:reluUnit];
	MPSCNNConvolution *conv1layer = [[MPSCNNConvolution alloc] initWithDevice:device convolutionDescriptor:conv1descriptor kernelWeights:conv1weights biasTerms:conv1biases flags:MPSCNNConvolutionFlagsNone];
	MPSImageDescriptor *conv1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide height:kImageSide featureChannels:32];

	MPSCNNPoolingMax *pool1layer = [[MPSCNNPoolingMax alloc] initWithDevice:device kernelWidth:2 kernelHeight:2 strideInPixelsX:2 strideInPixelsY:2];
	pool1layer.offset = (MPSOffset) { 1, 1, 0 };
	pool1layer.edgeMode = MPSImageEdgeModeClamp;
	MPSImageDescriptor *pool1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide2 height:kImageSide2 featureChannels:32];

	MPSCNNConvolutionDescriptor *conv2descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:5 kernelHeight:5 inputFeatureChannels:32 outputFeatureChannels:64 neuronFilter:reluUnit];
	MPSCNNConvolution *conv2layer = [[MPSCNNConvolution alloc] initWithDevice:device convolutionDescriptor:conv2descriptor kernelWeights:conv2weights biasTerms:conv2biases flags:MPSCNNConvolutionFlagsNone];
	MPSImageDescriptor *conv2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide2 height:kImageSide2 featureChannels:64];

	MPSCNNPoolingMax *pool2layer = [[MPSCNNPoolingMax alloc] initWithDevice:device kernelWidth:2 kernelHeight:2 strideInPixelsX:2 strideInPixelsY:2];
	pool2layer.offset = (MPSOffset) { 1, 1, 0 };
	pool2layer.edgeMode = MPSImageEdgeModeClamp;
	MPSImageDescriptor *pool2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:kImageSide4 height:kImageSide4 featureChannels:64];

	MPSCNNConvolutionDescriptor *fc1descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:kImageSide4 kernelHeight:kImageSide4 inputFeatureChannels:64 outputFeatureChannels:1024 neuronFilter:reluUnit];
	MPSCNNFullyConnected *fc1layer = [[MPSCNNFullyConnected alloc] initWithDevice:device convolutionDescriptor:fc1descriptor kernelWeights:fc1weights biasTerms:fc1biases flags:MPSCNNConvolutionFlagsNone];
	MPSImageDescriptor *fc1outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:1024];

	MPSCNNConvolutionDescriptor *fc2descriptor = [MPSCNNConvolutionDescriptor cnnConvolutionDescriptorWithKernelWidth:1 kernelHeight:1 inputFeatureChannels:1024 outputFeatureChannels:kOutputs neuronFilter:nil];
	MPSCNNFullyConnected *fc2layer = [[MPSCNNFullyConnected alloc] initWithDevice:device convolutionDescriptor:fc2descriptor kernelWeights:fc2weights biasTerms:fc2biases flags:MPSCNNConvolutionFlagsNone];
	MPSImageDescriptor *fc2outdescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:kOutputs];

	MPSImageDescriptor *softmaxOutput = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat16 width:1 height:1 featureChannels:kOutputs];
	MPSCNNSoftMax *softmaxLayer = [[MPSCNNSoftMax alloc] initWithDevice:device];

	MPSImageDescriptor *inputDescriptor = [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32 width:kImageSide height:kImageSide featureChannels:1];

	NSMutableArray<id<MTLCommandBuffer>> *pendingBuffers = [[NSMutableArray alloc] init];
	NSMutableArray<MPSImage *> *results = [[NSMutableArray alloc] init];

	const auto start = CACurrentMediaTime();

	for (size_t inputIndex = 0; inputIndex < kUsedExamples; inputIndex++) {
		id<MTLCommandBuffer> buffer = [queue commandBuffer];

		MPSImage *inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDescriptor];
		[inputImage.texture replaceRegion:MTLRegionMake2D(0, 0, kImageSide, kImageSide) mipmapLevel:0 withBytes:x + inputIndex * kInputLength bytesPerRow:sizeof(float) * kImageSide];

		[MPSTemporaryImage prefetchStorageWithCommandBuffer:buffer imageDescriptorList:@[conv1outdescriptor, pool1outdescriptor, conv2outdescriptor, pool2outdescriptor, fc1outdescriptor, fc2outdescriptor]];

		MPSTemporaryImage *c1o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:conv1outdescriptor];
		[conv1layer encodeToCommandBuffer:buffer sourceImage:inputImage destinationImage:c1o];

		MPSTemporaryImage *p1o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:pool1outdescriptor];
		[pool1layer encodeToCommandBuffer:buffer sourceImage:c1o destinationImage:p1o];

		MPSTemporaryImage *c2o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:conv2outdescriptor];
		[conv2layer encodeToCommandBuffer:buffer sourceImage:p1o destinationImage:c2o];

		MPSTemporaryImage *p2o = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:pool2outdescriptor];
		[pool2layer encodeToCommandBuffer:buffer sourceImage:c2o destinationImage:p2o];

		MPSTemporaryImage *fc1tdi = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:fc1outdescriptor];
		[fc1layer encodeToCommandBuffer:buffer sourceImage:p2o destinationImage:fc1tdi];

		MPSTemporaryImage *fc2tdi = [MPSTemporaryImage temporaryImageWithCommandBuffer:buffer imageDescriptor:fc2outdescriptor];
		[fc2layer encodeToCommandBuffer:buffer sourceImage:fc1tdi destinationImage:fc2tdi];

		MPSImage *resultsImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:softmaxOutput];
		[softmaxLayer encodeToCommandBuffer:buffer sourceImage:fc2tdi destinationImage:resultsImage];
		[results addObject:resultsImage];

		[buffer commit];
		[pendingBuffers addObject:buffer];
	}

	[pendingBuffers enumerateObjectsUsingBlock:^(id<MTLCommandBuffer> buffer, NSUInteger idx, BOOL *stop) {
		[buffer waitUntilCompleted];
	}];

	NSLog(@"Time: %g seconds", CACurrentMediaTime() - start);

	__block int correctExamples = 0;
	[pendingBuffers enumerateObjectsUsingBlock:^(id<MTLCommandBuffer> buffer, NSUInteger idx, BOOL *stop) {
		const size_t numSlices = (results[idx].featureChannels + 3)/4;
		float16_t halfs[numSlices * 4];
		for (size_t i = 0; i < numSlices; i += 1) {
			[results[idx].texture getBytes:&halfs[i * 4] bytesPerRow:8 bytesPerImage:8 fromRegion:MTLRegionMake3D(0, 0, 0, 1, 1, 1) mipmapLevel:0 slice:i];
		}

		float results[kOutputs];

		vImage_Buffer fullResultVImagebuf;
		fullResultVImagebuf.data = results;
		fullResultVImagebuf.height = 1;
		fullResultVImagebuf.width = kOutputs;
		fullResultVImagebuf.rowBytes = kOutputs * 4;

		vImage_Buffer halfResultVImagebuf;
		halfResultVImagebuf.data = halfs;
		halfResultVImagebuf.height = 1;
		halfResultVImagebuf.width = kOutputs;
		halfResultVImagebuf.rowBytes = kOutputs * 2;

		vImageConvert_Planar16FtoPlanarF(&halfResultVImagebuf, &fullResultVImagebuf, 0);

		int bestIndex = -1;
		float bestProbability = 0;
		for (auto i = 0; i < kOutputs; i++) {
			const auto probability = results[i];
			if (probability > bestProbability) {
				bestProbability = probability;
				bestIndex = i;
			}
		}

		if (bestIndex == expectedLabels[idx]) {
			correctExamples++;
		}
	}];

	NSLog(@"Accuracy: %f", static_cast<float>(correctExamples) / kUsedExamples);
}

@end
