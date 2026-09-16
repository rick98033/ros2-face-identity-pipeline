cmake_minimum_required(VERSION 3.8)
project(thor_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AuthorizedTarget.msg"
  "msg/CameraFrameTiming.msg"
  "msg/FaceIdentityCandidate.msg"
  "msg/FaceIdentityCandidates.msg"
  "msg/FaceLandmarks.msg"
  "msg/FaceTrack.msg"
  "msg/FaceTracks.msg"
  "msg/HealthStatus.msg"
  "msg/PerceptionMetrics.msg"
  "msg/PersonState.msg"
  "msg/PersonTrack.msg"
  "msg/PersonTracks.msg"
  "srv/CancelFollow.srv"
  "srv/GetCameraFrame.srv"
  "srv/GetFaceSnapshot.srv"
  "srv/GetPeopleSnapshot.srv"
  "srv/GetSceneSummary.srv"
  "srv/InferFaceId.srv"
  "srv/RequestFollow.srv"
  "srv/SelectFollowTarget.srv"
  DEPENDENCIES std_msgs builtin_interfaces vision_msgs geometry_msgs
)

ament_package()
