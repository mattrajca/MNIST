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

using namespace tensorflow;

static constexpr int kUsedExamples = 5000;
static constexpr int kImageSide = 28;
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

@end
