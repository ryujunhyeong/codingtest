# Single Saggar Detection Message
Header header

# Bounding box (x, y, width, height)
float32 x
float32 y
float32 width
float32 height

# Center point
geometry_msgs/Point center

# Detection confidence score
float32 confidence

# Layer and column information
int32 layer_id
int32 column_id

# Contour points (internal boundary)
geometry_msgs/Point[] contour_points

# Mask (optional, can be large)
sensor_msgs/Image mask
