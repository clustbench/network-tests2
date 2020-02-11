#pragma once

// Different error codes which can be returned from functions.
// File '../GUI/err_msgs.h' contains translated strings to all of the codes.

namespace NV {
typedef enum {
	Success=0, // no error

	NoMem,   // not enough memory
	
	CannotOpen, // cannot simply open input file
	
	NotANetCDF, // input file is not a NetCDF file
	
	UnexpEOF, // unexpected EOF in input text file
	
	NoNumProc, // cannot read number of processors
	NoBegMesLen, // cannot read begin message length
	NoEndMesLen, // cannot read end message length
	NoStepLen, // cannot read step length
	NoNoiseMesLen, // cannot read noise message length
	NoNoiseMesNum, // cannot read number of noise messages
	NoNoiseNumProc, // cannot read number of noise processors
	NoRpts, // cannot read number of repeats
	No3DData, // input file contains no data matrices
	
	DiffWdHght, // Width and height of 2D submatrices are not equal in input file
	
	NoHosts, // file with hosts' names is inaccessible
	
	IncmpDatDev, // data file and file with deviations are incompatible
	IncmpDat1Dat2, // two data files are incompatible
	
	InvRead, // trying to read data beyond the end of a file
	
	ErrNetCDFRead, // error while reading values from NetCDF file
	
	SameFileTwice, // the same file was selected twice
	
	NoViewer, // a viewer was not created
	
	UnknownMode, // unknown working mode
	
	/* specific for TabViewer */
	WndLoadedPartly, // not the whole desired set of matrices was loaded
	
	/* specific for FullViewer */
	NoRenderer, // cannot not create a renderer
	RenderError, // error occured in a renderer
	
	/* specific for TopologyViewer */
	NoTopoHosts, // hosts' names are undefined
	NoMatchingTopo, // input DOT file does not contain a matching topology
	DuplicateNames, // input DOT file contains duplicate names
	UselessTopoMatch, // two topologies match but it is useless
	
	/* specific for input files in DOT format */
	NoLexGraph, // missed 'graph'
	BraceNotLast, // '{' is not the last visible symbol in the line
	InvLexVertex, // vertex ID does not consist of letter 'v' followed by a number
	OverflIndVert, // overflow in a vertex index
	NoLexOpenBracket, // missed '['
	NoLexCloseBracket, // missed ']'
	NoLexLabel, // missed 'label='
	NoLexFrwQuote, // missed forward quote
	NoLexBckwQuote, // missed backward quote
	NoLexSemicolon, // missed ';'
	SemicolonNotLast, // ';' symbol is not at the end of the line
	OverflIndEdge, // overflow in an end of an edge
	InvVertInEdge, // incorrect vertex at one end of an edge was found
	NoEdgeMark, // missed '--'
	SelfLoopEdge, // self-loop was found
	ZeroBandw, // zero or negative bandwidth was found
	NoLexBits, // missed 'Gbit/s'
	DuplicateEdge, // duplicate edge was found
	
	LastCode // (do not use directly)
} ErrCode;
}

