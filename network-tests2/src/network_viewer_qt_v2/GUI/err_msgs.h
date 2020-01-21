#pragma once

#include "../core/err_codes.h"
#include <cstdarg>
#include <QObject>

class ErrMsgs {
  private:
	  // table of translated strings matching the error codes in '../core/err_codes.h';
	  // note that all the strings are in PLAIN format!
	  //
	  // rules for substitution of arguments:
	  //   1) placeholders must look like "%N", where 'N' is a number of substitution
	  //   2) only QString's are substituted
	  //
	  // for example, the result for "abcd %1 qwerty %2" can look like "abcd str qwerty string"
	  static QString err_msgs[];

  private:
	  // fills 'err_msgs'; it should be called only once
	  static void DefineErrMsgs (void);

  public:
	  // returns the message corresponding to error code 'c' after substituting the arguments;
	  //
	  // all arguments must be POINTERS to class QString (because of 'va_arg' specific);
	  //
	  // 'num_args' provides more control to the number of passed arguments
	  static QString ToString (const NV::ErrCode c, int num_args=0, ...);
};

//QString ErrMsgs::err_msgs[NV::LastCode]; - linker complains about "multiple definitions"; see it in mainwindow.cpp

inline void ErrMsgs::DefineErrMsgs (void) {
	QString inv_fmt=QObject::tr("file '%1' has incorrect format:");
	inv_fmt+="<br>&nbsp; &nbsp; &nbsp; "; // new line and 6 spaces
	
	//err_msgs[NV::Success]=O;
	err_msgs[NV::NoMem]=QObject::tr("not enough memory");
	err_msgs[NV::CannotOpen]=QObject::tr("file '%1' does not exist or cannot be opened");
	err_msgs[NV::NotANetCDF]=QObject::tr("file '%1' is not a valid NetCDF file");
	err_msgs[NV::UnexpEOF]=QObject::tr("unexpected end of file '%1'");
	(err_msgs[NV::NoNumProc]=inv_fmt)+=QObject::tr("cannot read number of processors");
	(err_msgs[NV::NoBegMesLen]=inv_fmt)+=QObject::tr("cannot read begin message length");
	(err_msgs[NV::NoEndMesLen]=inv_fmt)+=QObject::tr("cannot read end message length");
	(err_msgs[NV::NoStepLen]=inv_fmt)+=QObject::tr("cannot read step length");
	(err_msgs[NV::NoNoiseMesLen]=inv_fmt)+=QObject::tr("cannot read noise message length");
	(err_msgs[NV::NoNoiseMesNum]=inv_fmt)+=QObject::tr("cannot read number of noise messages");
	(err_msgs[NV::NoNoiseNumProc]=inv_fmt)+=QObject::tr("cannot read number of noise processors");
	(err_msgs[NV::NoRpts]=inv_fmt)+=QObject::tr("cannot read number of repeats");
	(err_msgs[NV::No3DData]=inv_fmt)+=QObject::tr("no 3D data was found");
	err_msgs[NV::DiffWdHght]=QObject::tr("2D submatrices are not square in the file '%1'");
	err_msgs[NV::NoHosts]=QObject::tr("file '%1' with hosts' names does not exist or cannot be opened");
	err_msgs[NV::IncmpDatDev]=QObject::tr("data file and file with deviations are incompatible");
	err_msgs[NV::IncmpDat1Dat2]=QObject::tr("two data files are incompatible");
	err_msgs[NV::InvRead]=QObject::tr("data read beyond the end of the file");
	err_msgs[NV::ErrNetCDFRead]=QObject::tr("error while reading NetCDF file");
	err_msgs[NV::SameFileTwice]=QObject::tr("the same file was selected twice");
	err_msgs[NV::NoViewer]=QObject::tr("the viewer was not created");
	err_msgs[NV::UnknownMode]=QObject::tr("unknown working mode");
	//err_msgs[NV::WndLoadedPartly]=O;
	err_msgs[NV::NoRenderer]=QObject::tr("cannot create the renderer: %1");
	err_msgs[NV::RenderError]=QObject::tr("error in the renderer: %1");
	err_msgs[NV::NoTopoHosts]=QObject::tr("hosts' names are undefined. Cannot proceed");
	err_msgs[NV::NoMatchingTopo]=QObject::tr("file '%1' does not contain a matching topology");
	err_msgs[NV::DuplicateNames]=QObject::tr("file '%1' contains duplicate names");
	err_msgs[NV::UselessTopoMatch]=QObject::tr("the topology from file '%1' and the retrieved topology match"
											   " but it is useless");
	(err_msgs[NV::NoLexGraph]=inv_fmt)+=QObject::tr("missed 'graph'");
	(err_msgs[NV::BraceNotLast]=inv_fmt)+=QObject::tr("'{' symbol is not the last visible symbol in the line");
	(err_msgs[NV::InvLexVertex]=inv_fmt)+=QObject::tr("a vertex ID must consist of letter 'v' followed by a number");
	(err_msgs[NV::OverflIndVert]=inv_fmt)+=QObject::tr("overflow in a vertex index");
	(err_msgs[NV::NoLexOpenBracket]=inv_fmt)+=QObject::tr("missed '['");
	(err_msgs[NV::NoLexCloseBracket]=inv_fmt)+=QObject::tr("missed ']'");
	(err_msgs[NV::NoLexLabel]=inv_fmt)+=QObject::tr("missed 'label='");
	(err_msgs[NV::NoLexFrwQuote]=inv_fmt)+=QObject::tr("missed forward quote");
	(err_msgs[NV::NoLexBckwQuote]=inv_fmt)+=QObject::tr("missed backward quote");
	(err_msgs[NV::NoLexSemicolon]=inv_fmt)+=QObject::tr("missed ';'");
	(err_msgs[NV::SemicolonNotLast]=inv_fmt)+=QObject::tr("';' symbol is not at the end of the line");
	(err_msgs[NV::OverflIndEdge]=inv_fmt)+=QObject::tr("overflow in an end of an edge");
	(err_msgs[NV::InvVertInEdge]=inv_fmt)+=QObject::tr("incorrect vertex at one end of an edge was found");
	(err_msgs[NV::NoEdgeMark]=inv_fmt)+=QObject::tr("missed '--'");
	(err_msgs[NV::SelfLoopEdge]=inv_fmt)+=QObject::tr("self-loop was found");
	(err_msgs[NV::ZeroBandw]=inv_fmt)+=QObject::tr("zero or negative bandwidth was found");
	(err_msgs[NV::NoLexBits]=inv_fmt)+=QObject::tr("missed 'Gbit/s'");
	(err_msgs[NV::DuplicateEdge]=inv_fmt)+=QObject::tr("duplicate edge was found");
}

inline QString ErrMsgs::ToString (const NV::ErrCode c, int num_args, ...) {
	int max_num=0; // number of substitutions (depending on error string)
	
	if (err_msgs[NV::NoMem].isEmpty()) DefineErrMsgs(); // if the table is empty then fill it
	
	switch (c)
	{
		case NV::CannotOpen:
		case NV::NotANetCDF:
		case NV::UnexpEOF:
		case NV::NoNumProc:
		case NV::NoBegMesLen:
		case NV::NoEndMesLen:
		case NV::NoStepLen:
		case NV::NoNoiseMesLen:
		case NV::NoNoiseMesNum:
		case NV::NoNoiseNumProc:
		case NV::NoRpts:
		case NV::No3DData:
		case NV::DiffWdHght:
		case NV::NoHosts:
		case NV::NoRenderer:
		case NV::RenderError:
		case NV::NoMatchingTopo:
		case NV::DuplicateNames:
		case NV::UselessTopoMatch:
		case NV::NoLexGraph:
		case NV::BraceNotLast:
		case NV::InvLexVertex:
		case NV::OverflIndVert:
		case NV::NoLexOpenBracket:
		case NV::NoLexCloseBracket:
		case NV::NoLexLabel:
		case NV::NoLexFrwQuote:
		case NV::NoLexBckwQuote:
		case NV::NoLexSemicolon:
		case NV::SemicolonNotLast:
		case NV::OverflIndEdge:
		case NV::InvVertInEdge:
		case NV::NoEdgeMark:
		case NV::SelfLoopEdge:
		case NV::ZeroBandw:
		case NV::NoLexBits:
		case NV::DuplicateEdge:
			max_num=1;
			break;
		default: return err_msgs[c]; // nothing to substitute
	}
	
	va_list ap; // pointer to unnamed arguments
	QString str(err_msgs[c]); // error string to process
	int num=1; // index of substitution
	QString find; // string to search for
	int ind_find=0; // position of 'find' in 'err_msg'
		
	va_start(ap,num_args); // initialize; 'num_args' is the last named argument
	num_args=(max_num<num_args)? max_num : num_args; // choose minimum for safety
	for ( ; num<=num_args; ++num)
	{
		(find='%')+=QString::number(num); // "%N"
		ind_find=str.indexOf(find,ind_find); // search
		const QString &vn=*(va_arg(ap,const QString*));
		str.replace(ind_find,find.length(),vn); // substitute!
		ind_find+=vn.length(); // move farther
	}
	va_end(ap);
	for ( ; num<=max_num; ++num) // erase the rest of placeholders
	{
		(find='%')+=QString::number(num);
		ind_find=str.indexOf(find,ind_find);
		str.replace(ind_find,find.length(),QString());
	}
	return str;
}

