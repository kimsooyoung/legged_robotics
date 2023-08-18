# Changelog

## [1.3.0] - 2023-02-01

### Removed
- Removed Quadruped class
- Removed dynamic control extension dependency
- Used omni.isaac.sensor classes for Contact and IMU sensors

## [1.2.2] - 2022-12-10

### Fixed
- Updated camera pipeline with writers

## [1.2.1] - 2022-11-03

### Fixed
- Incorrect viewport name issue
- Viewports not docking correctly

## [1.2.0] - 2022-08-30

### Changed
- Remove direct legacy viewport calls

## [1.1.2] - 2022-05-19

### Changed
- Updated unitree vision class to use OG ROS nodes
- Updated ROS1/ROS2 quadruped standalone samples to use OG ROS nodes

## [1.1.1] - 2022-05-15

### Fixed
- DC joint order change related fixes. 

## [1.1.0] - 2022-05-05

### Added
- added the ANYmal robot

## [1.0.2] - 2022-04-21

### Changed
- decoupled sensor testing from A1 and Go1 unit test
- fixed contact sensor bug in example and standalone

## [1.0.1] - 2022-04-20

### Changed
- Replaced find_nucleus_server() with get_assets_root_path()

## [1.0.0] - 2022-04-13

### Added
- quadruped class, unitree class (support both a1, go1), unitree vision class (unitree class with stereo cameras), and unitree direct class (unitree class that subscribe to external controllers)
- quadruped controllers
- documentations and unit tests
- quadruped standalone with ros 1 and ros 2 vio examples