

// Straight lines making up tubes would of course be great.

// How to get them:
// We still have to splinar create the intermediate positions. We'd choose the intermediate position in
// the intermeding plane, assuming that planes are kept where they are and not rumpled.

// If we create quadratic through the vertex by matching 3 positions, 2 of which from splinar fit, then
// we just ask when we look at a neighbour, what actually is the direction normal to the arc of the face between them.
// That can be inferred from some interpolant between the two curves.

// Basically the choice is either we do curves with splines, or we do lines between planes. 
// In this latter case we can still consider both sections coming away from the plane together, for Lap: the faces added 
// together can combine to look in a particular direction. We use neighbour movement towards its above or below neighbour
// to get near where we need to be looking.

// We DO NOT define what shape is made on the offset planes. We do assert that each tube is facing so that we only look
// at the "above" vertex for normal derivative. We assume that the x-section area on offset plane is average of the 2 in the planes.
// For vertcell volume, this gives us trapezoidally the area of each prism. We can ignore the intersection & gap between prisms initially and 
// assume that they cancel out.

// Perhaps we should start with linear segments and progress to curves.

// The linear segments join IN-PLANE with the known cross-sectional polygon. Even if this is not orthogonal to either segment.

// Volume = (0.75*x-section area + 0.25*x-section for same vertex below)*0.5*length of line to vertex below
//		  + (0.75*x-section area + 0.25*x-section for same vertex above)*0.5*length of line to vertex above

// That's incorrect because x-section area is the section for our own plane. We want to know what it is orthogonal to the
// direction of the line segment.
// So it's?

// We have to assume the x-section is stretched a predictable amount by the relative orientation of the section.
// ie if the segment moves (in its own chosen direction) 0.1 x-units for 1 normal to the dataplane,
// then stretch to our cross-section = sqrt( (1+10^2)/(10^2) ) or inverse to get orthogonal cross-section area.


// --------------------------------------------------
// If dataplane is not near centre of cell, we need to consider that when we apply move to it. Average "v_cell" with what
// we get below -- probably using a spline of some kind. Can develop initially without.

// What about for Lap phi?? 
// Where does our phi value theoretically exist. If in plane (yes) then how do we justify using the value from (average phi/ area)?
// The image is that we can model the "outward" derivative as represented here for the whole face...
// but back the other way?

// The conclusion is that if possible we should have equal extensions north and south. Not always possible...
// Have to live with it somehow. But bear in mind.
// --------------------------------------------------


// For uppermost angled plane we assume that cells are horizontal to left. (They subtend the final plane, the join is at the anode edge.)

// For linear approach to work, we have to assume that we have at least 22.5, 45, 67.5 degrees. Dutch insulator roof.
// 

// THAT IS ONE APPROACH - AND AS FAR AS WE KNOW IT *COULD* *WORK*.
// ---------------------

// vs

// The curve-based approach. This will assume that we create splinar vertex positions on each offset plane.
// There is a quadratic fit to the 3 positions (2 offset, 1 on).
// Neighbours also having a quadratic fit, we then estimate for the "face" towards a ptic neighbour, the
// integration of the normal vector over the face, assuming 
// 1. we can look in a plane containing the cell wall in the data-plane. 
// 2. therefore there are 2 curves and we take the average of them. Integrate up this
// but what is the width of cell wall effectively?

// That same question applies in linear (prism) case. 

// We can't do anything about it because we don't know that neighbour is even the same in the mesh above.
// We have to assume that for this cell, the cell wall lengths are what the dataplane suggests.
// 
// *** Of course,
//     some of them are being stretched by taking a section that is not orthogonal to the prism. ***
// We are in effect looking at a line diagonally across the wall/face when we look in-plane.
// But the transverse angles of the cells could be different!!

// So something complicated is then required. :-(
// ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

// Let's say it's about v carrying a species from one cell to another.
// We shall assume there is an "in-face vector" which is interpolated from the two tube-vectors.

// Different angles in to/from direction we can handle.

// Represent the face area :


// IDEA. We define the shape of the cell segmentfor the sake of definiteness.
// We define it by saying "NEIGHBOURS ARE SAME" BUT then we allow a distortion of the triangulation,
// a distortion of the polygons.
// Imagine a really big shear bend relative to another part of the mesh. Probably you get
// really tiny edges facing the nowdistant neighbour... IGNORE

// WLOG the dataplane is at z=z0. 
// We look in the plane and see a cell wall that apparently is going to show derivative in x-y.
// We know if the face were vertical the same then what we would get for v dot face,
// given the 'height' orthogonal to this plane that gets us to roughly the actual top of the face
// projected on to a plane transverse to the dataplane.

// We know if the face ALSO leans back, vxy effect is SAME, and we now have vz additionally an effect.
// 

// If face narrows --- including due to different angles of tubes --- then that changes the effect of vxy.
// But we could choose to ignore this -- the plane above has to handle the other half of the 
// distance to that plane.


// Meanwhile the transport to the cell above is determined by, obviously, v in that direction
// times the 50%-INTERPOLATED CROSS-SECTION -- ah so that does matter.

// Problem: if they lean differently away from each other, taking account of that for Lap, div etc
// This is inconsistent with the area used -- although if it comes from interpolated cross-section then the
// ap


// let's sort this mental bit out before anything else.

// The alternative: accept that we only have orthogonal cells for each plane. The tubes do not
// stay intact, but overlap certain others at the meeting point.
// Disadvantage: calculating overlaps, needing honeycomb?? And that things may flow a lot sideways and up into places we do not want
// instead of mesh being aligned to where it will probably flow. Current or fluid is trying to go diagonally but it goes into 2
// up and left instead. Those in turn end up with a lot of mass.



// THIS IS STUFF WE C A N THINK THROUGH AND GET RIGHT.

// Finding the


. Understand this --->
// Know we can make it work...
--->
// Add CUDA structs and do routine ..
