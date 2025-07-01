// Define points (starting at 9)
Point(9) = {1.25, -1.25, 0, 1};
Point(10) = {6, -1.25, 0, 1};
Point(11) = {6, 2, 0, 1};
Point(12) = {2, 2, 0, 1};
Point(13) = {-2, 2, 0, 1};
Point(14) = {-2, -2, 0, 1};
Point(15) = {-2, -6, 0, 1};
Point(16) = {1.25, -6, 0, 1};

// Create lines connecting the points (to form a closed curve)
Line(1) = {9, 10};
Line(2) = {10, 11};
Line(3) = {11, 12};
Line(4) = {12, 13};
Line(5) = {13, 14};
Line(6) = {14, 15};
Line(7) = {15, 16};
Line(8) = {16, 9};

// Number the sides by defining Physical Lines for each edge
Physical Line("Side 1") = {1};
Physical Line("Side 2") = {2};
Physical Line("Side 3") = {3};
Physical Line("Side 4") = {4};
Physical Line("Side 5") = {5};
Physical Line("Side 6") = {6};
Physical Line("Side 7") = {7};
Physical Line("Side 8") = {8};

// Form a closed loop from the lines
Line Loop(1) = {1, 2, 3, 4, 5, 6, 7, 8};

// Define a plane surface bounded by the loop
Plane Surface(1) = {1};

// Assign a physical group for the plane surface (if required)
Physical Surface("MySurface") = {1};

// Set the mesh size:  smaller value = finer mesh
Mesh.CharacteristicLengthMax = 0.20; // Adjust this value to change mesh fineness.