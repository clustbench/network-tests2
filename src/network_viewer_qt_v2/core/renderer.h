#pragma once

class QString;

enum PtRepr { CUBES, SPHERES, LIGHTS };

struct Coords { int x,y,z; };

/* All definitions and comments - in files 'renderer_OpenMP.*' 
   and 'renderer_OpenCL.*'. */

class Renderer {
  public:
	  virtual void Init (const unsigned int, const unsigned int, const int, const int, const int)=0;
	  virtual ~Renderer (void) {}
	  virtual const QString& GetError (void) const=0;
	  virtual unsigned int NumberCUs (void) const=0;
	  virtual void SetClrMatrix (unsigned short*)=0;
	  virtual void ShiftCamera (const float, const float, const float)=0;
	  virtual bool Zoom (const bool)=0;
	  virtual void Rotate (const float, const float)=0;
	  virtual void RotateOXY (const float, const bool)=0;
	  virtual void GetAxesForDrawing (float*) const=0;
	  virtual void RenderBox (unsigned int*) const=0;
	  virtual PtRepr ChangeKernel (const PtRepr)=0;
	  virtual void SetDepthConstraint (const unsigned int)=0;
	  virtual bool IsVolumeMode (void) const=0;
	  virtual bool ToggleVolumeMode (const bool)=0;
	  virtual bool BuildVolume (const unsigned char, const unsigned char, const unsigned char, const unsigned char)=0;
	  virtual void SelectPoints_by_click (const int, const int, const unsigned short *const, Coords*&, int&) const=0;
	  virtual void FreePoints (Coords*) const=0;
	  virtual bool SelectPoints_last_phase (const int, const int, Coords&, const unsigned short *const)=0;
	  virtual void TurnOffPtSelection (void)=0;
};

