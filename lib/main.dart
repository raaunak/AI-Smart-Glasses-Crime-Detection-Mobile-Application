import 'dart:async';
import 'dart:collection';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:camera/camera.dart';
import 'package:tflite_flutter/tflite_flutter.dart';

List<CameraDescription> cameras = [];

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    cameras = await availableCameras();
  } catch (e) {
    debugPrint('Error fetching cameras: $e');
  }
  runApp(const SmartGlassesApp());
}

class SmartGlassesApp extends StatelessWidget {
  const SmartGlassesApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0D1117),
      ),
      home: const DashboardScreen(),
    );
  }
}

// Blueprint for a single frame held in phone RAM
class BufferedFrame {
  final int frameNumber;
  final DateTime timestamp;
  final Uint8List yPlaneBytes;
  final int width;
  final int height;

  BufferedFrame({
    required this.frameNumber,
    required this.timestamp,
    required this.yPlaneBytes,
    required this.width,
    required this.height,
  });
}

// Blueprint for one detected person (bounding box + 17 skeletal joints)
class PoseDetection {
  final double confidence;
  final Rect box; // Coordinates scaled 0..640
  final List<Offset?> keypoints; // 17 body joints (null if not visible)

  PoseDetection({
    required this.confidence,
    required this.box,
    required this.keypoints,
  });
}

// Blueprint for one detected weapon/threat object
class ThreatDetection {
  final String label;
  final double confidence;
  final Rect box; // Coordinates scaled 0..640

  ThreatDetection({
    required this.label,
    required this.confidence,
    required this.box,
  });
}

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key});

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  CameraController? _controller;
  bool _isInitialized = false;
  bool _isMonitoring = false;
  String _errorMessage = '';

  // AI State Variables
  Interpreter? _poseInterpreter;
  Interpreter? _threatInterpreter;
  List<ThreatDetection> _currentThreats = [];
  static const List<String> _threatLabels = ['handgun', 'knife', 'weapon'];
  bool _isAiLoaded = false;
  bool _isBusyInferring = false;
  List<PoseDetection> _currentDetections = [];
  int _lastInferenceMs = 0;

  // FPS & Adaptive Burst State
  DateTime _lastCaptureTime = DateTime.fromMillisecondsSinceEpoch(0);
  int _targetFps = 2; // Default 2 FPS to save battery
  bool _isBurstMode = false;
  Timer? _burstTimer;

  // Rolling RAM Buffer (240 frames = 2 minutes at 2 FPS)
  static const int _maxBufferSize = 240;
  final Queue<BufferedFrame> _rollingBuffer = Queue<BufferedFrame>();

  // Live Telemetry Variables
  int _totalFramesCaptured = 0;
  int _framesDeletedForPrivacy = 0;
  int _cloudEscalations = 0;
  String _lastFrameInfo = 'None';

  // Live Event Log List (Newest messages appear at the bottom)
  final List<String> _eventLogs = [];

  @override
  void initState() {
    super.initState();
    _initializeCamera();
    _loadAiModel();
    _addLog('System initialized. Ready to start 2 FPS edge pipeline.');
  }

  void _addLog(String message) {
    final now = DateTime.now();
    final timeStr =
        '${now.hour.toString().padLeft(2, '0')}:${now.minute.toString().padLeft(2, '0')}:${now.second.toString().padLeft(2, '0')}';
    setState(() {
      _eventLogs.insert(0, '[$timeStr] $message');
      if (_eventLogs.length > 40) {
        _eventLogs.removeLast();
      }
    });
  }

  double _calculateRamUsageMb() {
    int totalBytes = 0;
    for (final frame in _rollingBuffer) {
      totalBytes += frame.yPlaneBytes.lengthInBytes;
    }
    return totalBytes / (1024 * 1024);
  }

  Future<void> _initializeCamera() async {
    if (cameras.isEmpty) {
      setState(() => _errorMessage = 'No cameras found.');
      return;
    }

    _controller = CameraController(
      cameras.first,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup: ImageFormatGroup.yuv420,
    );

    try {
      await _controller!.initialize();
      if (!mounted) return;
      setState(() => _isInitialized = true);
    } catch (e) {
      setState(() => _errorMessage = 'Camera error: $e');
    }
  }

  Future<void> _loadAiModel() async {
    try {
      // 1. Load Pose Model
      final poseInterp = await Interpreter.fromAsset(
        'assets/models/yolo26n-pose.tflite',
      );
      // 2. Load Custom Threat Model
      final threatInterp = await Interpreter.fromAsset(
        'assets/models/best_yolo26n_threat.tflite',
      );

      final poseOut = poseInterp.getOutputTensor(0).shape;
      final threatOut = threatInterp.getOutputTensor(0).shape;

      setState(() {
        _poseInterpreter = poseInterp;
        _threatInterpreter = threatInterp;
        _isAiLoaded = true;
      });

      _addLog('🧠 Dual AI Ready! Pose: $poseOut | Threat: $threatOut');
    } catch (e) {
      _addLog('❌ AI Load Error: $e');
    }
  }

  Future<void> _runPoseInference(CameraImage image) async {
    if (_poseInterpreter == null || _isBusyInferring) return;
    _isBusyInferring = true;

    final stopwatch = Stopwatch()..start();

    try {
      // 1. Build the [1, 3, 640, 640] NCHW input buffer from the camera's Y-plane
      final Float32List inputBuffer = Float32List(1 * 3 * 640 * 640);
      final Uint8List yBytes = image.planes[0].bytes;
      final int srcW = image.width;
      final int srcH = image.height;
      final int yRowStride = image.planes[0].bytesPerRow;

      const int planeSize = 640 * 640;
      for (int y = 0; y < 640; y++) {
        // Rotate 90 degrees clockwise to match portrait phone camera orientation
        final int srcX = (y * srcW ~/ 640).clamp(0, srcW - 1);
        for (int x = 0; x < 640; x++) {
          final int srcY = ((639 - x) * srcH ~/ 640).clamp(0, srcH - 1);
          final int pixelIndex = srcY * yRowStride + srcX;
          final double normVal = yBytes[pixelIndex] / 255.0;

          final int outIdx = y * 640 + x;
          inputBuffer[outIdx] = normVal; // Red plane
          inputBuffer[planeSize + outIdx] = normVal; // Green plane
          inputBuffer[2 * planeSize + outIdx] = normVal; // Blue plane
        }
      }

      // 2. Prepare the [1, 56, 8400] output tensor
      final output = List.generate(
        1,
        (_) => List.generate(56, (_) => List<double>.filled(8400, 0.0)),
      );

      // 3. Run inference
      _poseInterpreter!.run(inputBuffer.buffer, output);

      // 4. Parse detections from the 8,400 candidates
      final features = output[0];
      final List<PoseDetection> found = [];

      for (int i = 0; i < 8400; i++) {
        final double conf = features[4][i];
        if (conf < 0.45) continue;

        double cx = features[0][i];
        double cy = features[1][i];
        double w = features[2][i];
        double h = features[3][i];

        // Scale normalized coordinates (0..1) to 640x640 if needed
        if (cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5) {
          cx *= 640;
          cy *= 640;
          w *= 640;
          h *= 640;
        }

        final rect = Rect.fromLTRB(
          cx - w / 2,
          cy - h / 2,
          cx + w / 2,
          cy + h / 2,
        );

        // Skip overlapping duplicate boxes
        final bool overlaps = found.any((d) => d.box.overlaps(rect));
        if (overlaps) continue;

        // Extract the 17 skeletal joints (51 numbers starting at index 5)
        final List<Offset?> kpts = [];
        for (int k = 0; k < 17; k++) {
          double kx = features[5 + k * 3][i];
          double ky = features[5 + k * 3 + 1][i];
          final double kconf = features[5 + k * 3 + 2][i];

          if (kx <= 1.5 && ky <= 1.5) {
            kx *= 640;
            ky *= 640;
          }
          kpts.add(kconf > 0.4 ? Offset(kx, ky) : null);
        }

        found.add(PoseDetection(confidence: conf, box: rect, keypoints: kpts));
        if (found.length >= 3) break;
      }

      // 5. Run Custom Threat Object Detection on the same 640x640 inputBuffer
      final List<ThreatDetection> foundThreats = [];
      if (_threatInterpreter != null) {
        final tShape = _threatInterpreter!.getOutputTensor(0).shape;
        // tShape is typically [1, 7, 8400] (4 box coords + 3 classes) or [1, 300, 6]
        final int dim1 = tShape[1];
        final int dim2 = tShape[2];

        final threatOutput = List.generate(
          1,
          (_) => List.generate(dim1, (_) => List<double>.filled(dim2, 0.0)),
        );

        _threatInterpreter!.run(inputBuffer.buffer, threatOutput);
        final tMatrix = threatOutput[0];

        // Case A: Standard YOLO shape [1, 4 + numClasses, 8400]
        if (dim2 == 8400) {
          final int numClasses = dim1 - 4;
          for (int i = 0; i < 8400; i++) {
            double maxConf = 0.0;
            int bestClass = 0;
            for (int c = 0; c < numClasses; c++) {
              final double score = tMatrix[4 + c][i];
              if (score > maxConf) {
                maxConf = score;
                bestClass = c;
              }
            }
            if (maxConf < 0.40) continue;

            double cx = tMatrix[0][i];
            double cy = tMatrix[1][i];
            double w = tMatrix[2][i];
            double h = tMatrix[3][i];
            if (cx <= 1.5 && cy <= 1.5 && w <= 1.5 && h <= 1.5) {
              cx *= 640;
              cy *= 640;
              w *= 640;
              h *= 640;
            }
            final rect = Rect.fromLTRB(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2);
            if (foundThreats.any((t) => t.box.overlaps(rect))) continue;

            final label = bestClass < _threatLabels.length
                ? _threatLabels[bestClass]
                : 'threat_$bestClass';
            foundThreats.add(ThreatDetection(label: label, confidence: maxConf, box: rect));
            if (foundThreats.length >= 3) break;
          }
        }
        // Case B: YOLO26 End-to-End NMS-Free shape [1, 300, 6] ([x1, y1, x2, y2, conf, classId])
        else if (dim2 == 6) {
          for (int i = 0; i < dim1; i++) {
            final double conf = tMatrix[i][4];
            if (conf < 0.40) continue;
            double x1 = tMatrix[i][0];
            double y1 = tMatrix[i][1];
            double x2 = tMatrix[i][2];
            double y2 = tMatrix[i][3];
            final int clsId = tMatrix[i][5].round();
            if (x2 <= 1.5 && y2 <= 1.5) {
              x1 *= 640;
              y1 *= 640;
              x2 *= 640;
              y2 *= 640;
            }
            final label = clsId < _threatLabels.length
                ? _threatLabels[clsId]
                : 'threat_$clsId';
            foundThreats.add(
              ThreatDetection(
                label: label,
                confidence: conf,
                box: Rect.fromLTRB(x1, y1, x2, y2),
              ),
            );
          }
        }
      }

      stopwatch.stop();
      if (!mounted) return;

      setState(() {
        _currentDetections = found;
        _currentThreats = foundThreats;
        _lastInferenceMs = stopwatch.elapsedMilliseconds;
      });

      if (foundThreats.isNotEmpty) {
        final topThreat = foundThreats.first;
        _addLog(
          '🚨 WEAPON DETECTED: ${topThreat.label.toUpperCase()} (${(topThreat.confidence * 100).toStringAsFixed(0)}%) in ${_lastInferenceMs}ms',
        );
        _triggerAdaptiveBurst();
      }

      if (found.isNotEmpty) {
        _addLog(
          '🧍 Pose Detected: ${found.length} person(s) (${(found.first.confidence * 100).toStringAsFixed(0)}%) in ${_lastInferenceMs}ms',
        );
        _evaluatePoseThreat(found);
      }
    } catch (e) {
      _addLog('⚠️ Inference error: $e');
    } finally {
      _isBusyInferring = false;
    }
  }

  // Checks if any detected skeleton shows a threat/distress posture
  void _evaluatePoseThreat(List<PoseDetection> detections) {
    for (final det in detections) {
      final leftShoulder = det.keypoints[5];
      final rightShoulder = det.keypoints[6];
      final leftWrist = det.keypoints[9];
      final rightWrist = det.keypoints[10];

      // Check if Left or Right Wrist is raised higher than the Shoulder (Y is smaller at the top)
      final bool leftRaised = leftShoulder != null &&
          leftWrist != null &&
          leftWrist.dy < leftShoulder.dy;

      final bool rightRaised = rightShoulder != null &&
          rightWrist != null &&
          rightWrist.dy < rightShoulder.dy;

      // 1. Both hands raised above shoulders (Hold-up / Surrender / Overhead strike)
      if (leftRaised && rightRaised) {
        _addLog('🚨 POSE ANOMALY: Hands raised above shoulders detected!');
        _triggerAdaptiveBurst();
        return;
      }

      // 2. Rapid proximity / Lunge check (Bounding box covers >65% of vertical frame)
      if (det.box.height > (640 * 0.65) && (leftRaised || rightRaised)) {
        _addLog('🚨 POSE ANOMALY: Close-range aggressive stance detected!');
        _triggerAdaptiveBurst();
        return;
      }
    }
  }

  Future<void> _toggleMonitoring() async {
    if (_controller == null || !_isInitialized) return;

    if (_isMonitoring) {
      await _controller!.stopImageStream();
      _burstTimer?.cancel();
      setState(() {
        _isMonitoring = false;
        _isBurstMode = false;
        _targetFps = 2;
        _currentDetections = [];
      });
      _addLog(
        '🛑 Pipeline stopped. RAM buffer paused at ${_rollingBuffer.length} frames.',
      );
    } else {
      setState(() => _isMonitoring = true);
      _addLog(
        '🟢 Pipeline started: Sampling camera at 2 FPS (500ms interval).',
      );

      await _controller!.startImageStream((CameraImage image) {
        final now = DateTime.now();
        final int intervalMs = (1000 / _targetFps).round();

        // 2 FPS Gatekeeper
        if (now.difference(_lastCaptureTime).inMilliseconds < intervalMs) {
          return;
        }
        _lastCaptureTime = now;

        // Copy raw frame bytes into phone RAM
        final Uint8List frameBytes = Uint8List.fromList(image.planes[0].bytes);
        final int sizeKb = (frameBytes.lengthInBytes / 1024).round();

        setState(() {
          _totalFramesCaptured++;
          _lastFrameInfo =
              'Frame #$_totalFramesCaptured (${image.width}x${image.height} • $sizeKb KB • ${_lastInferenceMs}ms)';

          // Rolling Buffer FIFO Queue
          if (_rollingBuffer.length >= _maxBufferSize) {
            _rollingBuffer.removeFirst();
            _framesDeletedForPrivacy++;
          }

          _rollingBuffer.addLast(
            BufferedFrame(
              frameNumber: _totalFramesCaptured,
              timestamp: now,
              yPlaneBytes: frameBytes,
              width: image.width,
              height: image.height,
            ),
          );
        });

        // Run live TFLite pose inference on this 2 FPS frame
        _runPoseInference(image);

        // Checks if any detected skeleton shows a threat/distress posture

        if (_totalFramesCaptured % 20 == 0) {
          _addLog(
            '📦 RAM Buffer holds ${_rollingBuffer.length} frames (${_calculateRamUsageMb().toStringAsFixed(1)} MB in RAM).',
          );
        }
      });
    }
  }

  void _triggerAdaptiveBurst() {
    if (!_isMonitoring || _isBurstMode) return;

    setState(() {
      _isBurstMode = true;
      _targetFps = 10;
      _cloudEscalations++;
    });

    _addLog('⚠️ SUSPICION TRIGGERED: Escalating to 10 FPS Burst Mode for 3s!');

    _burstTimer?.cancel();
    _burstTimer = Timer(const Duration(seconds: 3), () {
      if (!mounted || !_isMonitoring) return;
      setState(() {
        _isBurstMode = false;
        _targetFps = 2;
      });
      _addLog(
        '☁️ Burst complete. Frozen ${_rollingBuffer.length} frames (${_calculateRamUsageMb().toStringAsFixed(1)} MB) ready for Backend. Reverted to 2 FPS.',
      );
    });
  }

  @override
  void dispose() {
    _poseInterpreter?.close();
    _threatInterpreter?.close();
    _burstTimer?.cancel();
    _controller?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final double ramMb = _calculateRamUsageMb();

    return Scaffold(
      appBar: AppBar(
        title: const Text(
          'SMART GLASSES EDGE DASHBOARD',
          style: TextStyle(fontSize: 15, fontWeight: FontWeight.bold),
        ),
        backgroundColor: const Color(0xFF161B22),
        centerTitle: true,
      ),
      body: Column(
        children: [
          // ZONE 1: SYSTEM STATUS CHIPS
          Container(
            padding: const EdgeInsets.symmetric(vertical: 8, horizontal: 12),
            color: const Color(0xFF161B22),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                _buildChip(
                  Icons.psychology,
                  _isAiLoaded
                      ? 'AI: Ready (${_lastInferenceMs}ms)'
                      : 'AI: Loading',
                  _isAiLoaded ? Colors.cyanAccent : Colors.grey,
                ),
                _buildChip(
                  Icons.speed,
                  _isBurstMode ? 'BURST: 10 FPS' : 'WATCH: 2 FPS',
                  _isBurstMode ? Colors.redAccent : Colors.greenAccent,
                ),
                _buildChip(
                  Icons.memory,
                  'RAM: ${ramMb.toStringAsFixed(1)} MB',
                  Colors.amberAccent,
                ),
              ],
            ),
          ),

          // ZONE 2: METRICS BAR
          Container(
            padding: const EdgeInsets.symmetric(vertical: 10, horizontal: 8),
            decoration: const BoxDecoration(
              color: Color(0xFF0D1117),
              border: Border(bottom: BorderSide(color: Colors.white12)),
            ),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.spaceAround,
              children: [
                _buildStat('Captured', '$_totalFramesCaptured', Colors.white),
                _buildStat(
                  'RAM Cache',
                  '${_rollingBuffer.length}/$_maxBufferSize',
                  Colors.greenAccent,
                ),
                _buildStat(
                  'Privacy Wiped',
                  '$_framesDeletedForPrivacy',
                  Colors.cyanAccent,
                ),
                _buildStat(
                  'Escalations',
                  '$_cloudEscalations',
                  Colors.orangeAccent,
                ),
              ],
            ),
          ),

          // ZONE 3: LIVE CAMERA VIEWFINDER + SKELETON OVERLAY
          Expanded(
            flex: 5,
            child: Container(
              width: double.infinity,
              color: Colors.black,
              child: _buildCameraStage(),
            ),
          ),

          // ZONE 4: LIVE EVENT LOG (Scrollable Dashboard Feed)
          Expanded(
            flex: 4,
            child: Container(
              padding: const EdgeInsets.all(12),
              color: const Color(0xFF161B22),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'LIVE PIPELINE TELEMETRY LOG',
                        style: TextStyle(
                          fontSize: 12,
                          fontWeight: FontWeight.bold,
                          color: Colors.cyanAccent,
                        ),
                      ),
                      Flexible(
                        child: Text(
                          _lastFrameInfo,
                          overflow: TextOverflow.ellipsis,
                          style: const TextStyle(
                            fontSize: 11,
                            color: Colors.white54,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const Divider(color: Colors.white24, height: 12),
                  Expanded(
                    child: ListView.builder(
                      reverse: true,
                      itemCount: _eventLogs.length,
                      itemBuilder: (context, index) {
                        final log = _eventLogs[index];
                        final isWarning = log.contains('⚠️');
                        final isPose = log.contains('🧍');
                        return Padding(
                          padding: const EdgeInsets.symmetric(vertical: 3),
                          child: Text(
                            log,
                            style: TextStyle(
                              fontSize: 12,
                              fontFamily: 'monospace',
                              color: isWarning
                                  ? Colors.redAccent
                                  : (isPose
                                        ? Colors.greenAccent
                                        : Colors.white70),
                              fontWeight: (isWarning || isPose)
                                  ? FontWeight.bold
                                  : FontWeight.normal,
                            ),
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),
          ),

          // ZONE 5: ACTION CONTROLS
          Container(
            padding: const EdgeInsets.all(12),
            color: const Color(0xFF0D1117),
            child: Row(
              children: [
                Expanded(
                  child: ElevatedButton(
                    onPressed: _toggleMonitoring,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isMonitoring
                          ? Colors.redAccent
                          : Colors.teal,
                      padding: const EdgeInsets.symmetric(vertical: 14),
                    ),
                    child: Text(
                      _isMonitoring ? 'STOP PIPELINE' : 'START 2 FPS PIPELINE',
                      style: const TextStyle(
                        fontWeight: FontWeight.bold,
                        color: Colors.white,
                      ),
                    ),
                  ),
                ),
                const SizedBox(width: 10),
                ElevatedButton(
                  onPressed: _isMonitoring ? _triggerAdaptiveBurst : null,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.orange[800],
                    padding: const EdgeInsets.symmetric(
                      vertical: 14,
                      horizontal: 14,
                    ),
                  ),
                  child: const Text(
                    '⚡ TEST BURST',
                    style: TextStyle(
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCameraStage() {
    if (_errorMessage.isNotEmpty) {
      return Center(
        child: Text(_errorMessage, style: const TextStyle(color: Colors.red)),
      );
    }
    if (!_isInitialized || _controller == null) {
      return const Center(
        child: CircularProgressIndicator(color: Colors.cyanAccent),
      );
    }

    return Stack(
      alignment: Alignment.center,
      children: [
        AspectRatio(
          aspectRatio: _controller!.value.aspectRatio,
          child: Stack(
            fit: StackFit.expand,
            children: [
              CameraPreview(_controller!),
              CustomPaint(painter: SkeletonPainter(_currentDetections, _currentThreats)),
            ],
          ),
        ),
        if (_isMonitoring)
          Positioned(
            top: 10,
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 5),
              decoration: BoxDecoration(
                color: _isBurstMode
                    ? Colors.red.withValues(alpha: 0.85)
                    : Colors.black87,
                borderRadius: BorderRadius.circular(16),
                border: Border.all(
                  color: _isBurstMode ? Colors.white : Colors.cyanAccent,
                ),
              ),
              child: Text(
                _isBurstMode
                    ? '⚡ ADAPTIVE BURST: 10 FPS (100ms)'
                    : '● EDGE SAMPLING: 2 FPS (500ms)',
                style: const TextStyle(
                  color: Colors.white,
                  fontSize: 11,
                  fontWeight: FontWeight.bold,
                ),
              ),
            ),
          ),
      ],
    );
  }

  Widget _buildChip(IconData icon, String text, Color color) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: Colors.white10,
        borderRadius: BorderRadius.circular(6),
      ),
      child: Row(
        children: [
          Icon(icon, size: 13, color: color),
          const SizedBox(width: 4),
          Text(
            text,
            style: TextStyle(
              fontSize: 11,
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStat(String label, String value, Color color) {
    return Column(
      children: [
        Text(
          value,
          style: TextStyle(
            fontSize: 17,
            fontWeight: FontWeight.bold,
            color: color,
          ),
        ),
        Text(
          label,
          style: const TextStyle(fontSize: 10, color: Colors.white60),
        ),
      ],
    );
  }
}

// Draws the green bounding box and cyan skeleton joints over the camera preview
class SkeletonPainter extends CustomPainter {
  final List<PoseDetection> detections;
  final List<ThreatDetection> threats;
  SkeletonPainter(this.detections, this.threats);

  static const List<List<int>> bones = [
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Shoulders, Elbows, Wrists
    [5, 11], [6, 12], [11, 12], // Torso / Hips
    [11, 13], [13, 15], [12, 14], [14, 16], // Knees & Ankles
  ];

  @override
  void paint(Canvas canvas, Size size) {
    final boxPaint = Paint()
      ..color = Colors.greenAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final bonePaint = Paint()
      ..color = Colors.cyanAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.5;

    final jointPaint = Paint()
      ..color = Colors.amberAccent
      ..style = PaintingStyle.fill;

    final double scaleX = size.width / 640.0;
    final double scaleY = size.height / 640.0;

    for (final det in detections) {
      final scaledBox = Rect.fromLTRB(
        det.box.left * scaleX,
        det.box.top * scaleY,
        det.box.right * scaleX,
        det.box.bottom * scaleY,
      );
      canvas.drawRect(scaledBox, boxPaint);

      for (final bone in bones) {
        final p1 = det.keypoints[bone[0]];
        final p2 = det.keypoints[bone[1]];
        if (p1 != null && p2 != null) {
          canvas.drawLine(
            Offset(p1.dx * scaleX, p1.dy * scaleY),
            Offset(p2.dx * scaleX, p2.dy * scaleY),
            bonePaint,
          );
        }
      }

      for (final kp in det.keypoints) {
        if (kp != null) {
          canvas.drawCircle(
            Offset(kp.dx * scaleX, kp.dy * scaleY),
            4.0,
            jointPaint,
          );
        }
      }
    }

    final threatPaint = Paint()
      ..color = Colors.redAccent
      ..style = PaintingStyle.stroke
      ..strokeWidth = 3.0;

    for (final threat in threats) {
      final scaledBox = Rect.fromLTRB(
        threat.box.left * scaleX,
        threat.box.top * scaleY,
        threat.box.right * scaleX,
        threat.box.bottom * scaleY,
      );
      canvas.drawRect(scaledBox, threatPaint);

      final textSpan = TextSpan(
        text: ' ${threat.label.toUpperCase()} ${(threat.confidence * 100).toStringAsFixed(0)}% ',
        style: const TextStyle(
          color: Colors.white,
          backgroundColor: Colors.redAccent,
          fontSize: 12,
          fontWeight: FontWeight.bold,
        ),
      );
      final textPainter = TextPainter(
        text: textSpan,
        textDirection: TextDirection.ltr,
      )..layout();
      textPainter.paint(canvas, Offset(scaledBox.left, scaledBox.top - 18));
    }
  }

  @override
  bool shouldRepaint(covariant CustomPainter oldDelegate) => true;
}
