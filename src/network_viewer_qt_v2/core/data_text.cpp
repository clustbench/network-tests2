#include "data_text.h"
#include <cstring>
#include <cctype>
#include <clocale>

namespace {
const char *const NEW_MESSAGE_LENGTH_STR="Message length ";
const size_t NEW_MESSAGE_LENGTH_STR_len=strlen(NEW_MESSAGE_LENGTH_STR);
}

bool Data_Text::readline (FILE *f, Line &l) {
	static const size_t buf_sz=256u;
	char buf[buf_sz];
	unsigned int len,len1,m;
	char *ln;
	
	if (fgets(buf,buf_sz,f)==NULL)
		return false;
	len=strlen(buf);
	if (l.mem_sz<=len)
	{
		if (l.line!=NULL) free(l.line);
		l.mem_sz=len+(len>>1u)+1u; // additionally reserve (len/2) bytes
		l.line=static_cast<char*>(malloc(l.mem_sz*sizeof(char)));
		if (l.line==NULL)
		{
			l.mem_sz=0u;
			return false;
		}
	}
	strcpy(l.line,buf);
	if (buf[len-1u]=='\n')
		return true;
	while (fgets(buf,buf_sz,f)!=NULL)
	{
		len1=strlen(buf);
		len+=len1;
		if (l.mem_sz<=len)
		{
			m=len+(len>>1u)+1u;
			ln=static_cast<char*>(malloc(m*sizeof(char)));
			if (ln==NULL) return false;
			strcpy(ln,l.line);
			free(l.line);
			l.line=ln;
			l.mem_sz=m;
		}
		strcpy(l.line+(len-len1),buf);
		if (l.line[len-1u]=='\n')
			break;
	}
	return true;
}

Data_Text::Data_Text (const QString &f_name, NV::ErrCode &err)
{
	this->source_fname=f_name;

	source_file=fopen(f_name.toLocal8Bit().constData()/*written so to process non-English paths*/,"r");
	if (source_file==NULL)
	{
		err=NV::CannotOpen;
		return;
	}
	
	Line l;
	const char *work_line;

	#define GETNEXTLINE \
			for ( ; ; ) \
			{ \
				if (!readline(source_file,l)) \
				{ \
					err=NV::UnexpEOF; \
					return; \
				} \
				if (!l.isallws()) break; \
			}

	#define READVAR(comment,offs,var,error) \
			GETNEXTLINE \
			work_line=strstr(l.Give(),comment); \
			if ((work_line==NULL) || (sscanf(work_line+offs,"%d",&var)<1)) \
			{ \
				err=NV:: error; \
				return; \
			}

	READVAR("processors ",11,this->num_processors,NoNumProc)
	GETNEXTLINE

	work_line=strstr(l.Give(),"test type ");
	if (work_line!=NULL)
	{
		work_line+=10; // adding length of string "test type "
		this->test_type=work_line;
	}
	
	GETNEXTLINE
	
	work_line=strstr(l.Give(),"data type ");
	if (work_line!=NULL)
	{
		work_line+=10; // adding length of string "data type "
		this->data_type=work_line;
	}
	
	READVAR("begin message length ",21,this->begin_message_length,NoBegMesLen)
	READVAR("end message length ",19,this->end_message_length,NoEndMesLen)
	READVAR("step length ",12,this->step_length,NoStepLen)
	READVAR("noise message length ",21,this->noise_message_length,NoNoiseMesLen)
	READVAR("number of noise messages ",25,this->noise_message_num,NoNoiseMesNum)
	READVAR("number of noise processes ",26,this->noise_processors,NoNoiseNumProc)
	READVAR("number of repeates ",19,this->num_repeats,NoRpts)
	GETNEXTLINE

	if (strstr(l.Give(),"hosts:")!=NULL)
	{
		GETNEXTLINE
	}
	
	unsigned int host_num=1u;
	QString name;
	char *line;
	
	line=(char*)malloc((NEW_MESSAGE_LENGTH_STR_len+11u)*sizeof(char));
	if (line==NULL)
	{
		err=NV::NoMem;
		return;
	}
	// 'line' will contain the very beginning of main data
	sprintf(line,"%s%d",NEW_MESSAGE_LENGTH_STR,begin_message_length);
	while (strstr(l.Give(),line)==NULL)
	{
		name.setNum(host_num);
		(name+=") ")+=l.Give(); // enumerated host's name
		this->host_names.push_back(name);
		++host_num;
		for ( ; ; )
		{
			if (!readline(source_file,l))
			{
				err=NV::No3DData;
				free(line);
				return;
			}
			if (!l.isallws()) break;
		}
	}
	free(line);
	
	fgetpos(source_file,&data_pos); // 'data_pos' is set!
	
	flt_pt=*(localeconv()->decimal_point);
	
	this->z_num=1;
	while (readline(source_file,l))
	{
		if (strstr(l.Give(),NEW_MESSAGE_LENGTH_STR)!=NULL)
			++(this->z_num);
	}
	
	matr_ind=row_val_ind=this->z_num;
	
	err=NV::Success;
}

void Data_Text::Begin (const IData::Portion p, const int mes_len) {
	Line l;
	const char *tmp;
	int m_l;
	
	matr_ind=row_val_ind=0;
	portion=p;
	fsetpos(source_file,&data_pos);
	if (mes_len!=this->begin_message_length)
	{
		matr_ind=(this->step_length<1)? 0 : ((mes_len-this->begin_message_length)/this->step_length);
		
		for ( ; ; )
		{
			if (!readline(source_file,l))
			{
				matr_ind=this->z_num; // further calls to GetDataAndMove() will return 'NV::InvRead' error
				return;
			}
			tmp=strchr(l.Give(),'M');
			if ((tmp!=NULL) && !strncmp(tmp,NEW_MESSAGE_LENGTH_STR,NEW_MESSAGE_LENGTH_STR_len))
			{
				tmp+=NEW_MESSAGE_LENGTH_STR_len;
				m_l=0;
				static_cast<void>(sscanf(tmp,"%d",&m_l));
				if (m_l==mes_len) break;
			}
		}
	}
}

NV::ErrCode Data_Text::GetDataAndMove (double *buf) {
	if (matr_ind>=this->z_num) return NV::InvRead; // Begin() was not called before!
	
	const int num1=this->num_processors-1;
	const int matr_size=(num1+1)*(num1+1);
	const char non_flt_pt=(flt_pt=='.')? ',' : '.';
	Line l;
	char *line,*tmp;
	int i=0,i2;
	
	#define GET_ROW \
			if (!readline(source_file,l)) return NV::UnexpEOF;\
			line=l.Give_mdf();\
			for (i2=i+num1; i<i2; ++i)\
			{\
				if ((tmp=strchr(line,non_flt_pt))!=NULL)\
					*tmp=flt_pt;\
				*buf=strtod(line,&tmp);\
				++buf;\
				line=tmp+1;\
			}\
			if ((tmp=strchr(line,non_flt_pt))!=NULL)\
				*tmp=flt_pt;\
			*buf=atof(line);\
			++buf;
	
	#define GO \
			while (readline(source_file,l))\
			{\
				if (strstr(l.Give(),NEW_MESSAGE_LENGTH_STR)!=NULL) break;\
			}
	
	switch (portion)
	{
		case IData::Value:
		{
			// remember that we have very small amount of dynamic memory!
			
			static const int str_len=64;
			char str[str_len+1]; // stack allocation
			int c;
			static const size_t symb_num=16u;
			
			i=0;
			while (i<str_len)
			{
				c=fgetc(source_file);
				if (c==EOF) return NV::UnexpEOF;
				if ((c=='\t') || (c=='\n')) break;
				str[i]=static_cast<char>(c);
				++i;
			}
			str[i]='\0';
			if ((tmp=strchr(str,non_flt_pt))!=NULL)
				*tmp=flt_pt;
			buf[0]=atof(str);
			++row_val_ind;
			if (row_val_ind==matr_size)
			{
				++matr_ind;
				row_val_ind=0;
				if (matr_ind!=this->z_num)
				{
					for ( ; ; )
					{
						if (fgets(str,symb_num,source_file)==NULL) return NV::UnexpEOF;
						if (!strcmp(str,NEW_MESSAGE_LENGTH_STR))
						{
							for ( ; ; )
							{
								if (fgets(str,symb_num,source_file)==NULL) return NV::UnexpEOF;
								if (strchr(str,'\n')!=NULL) break;
							}
							break;
						}
					}
				}
			}
			break;
		}
		case IData::Row:
			GET_ROW
			if (row_val_ind==num1)
			{
				row_val_ind=0;
				++matr_ind;
				if (matr_ind!=this->z_num)
				{
					GO
				}
			}
			else ++row_val_ind;
			break;
		case IData::Matrix:
		case IData::File:
			for ( ; ; )
			{
				while (i<matr_size)
				{
					GET_ROW
					++i;
				}
				++matr_ind;
				if (matr_ind==this->z_num) break;
				GO
				if (portion==IData::Matrix) break;
			}
			break;
	}
	return NV::Success;
}

NV::ErrCode Data_Text::GetSingleValue (const int mes_len, const int row, const int col, double &v)
{
	if ((mes_len<0) || (row<0) || (col<0) || (mes_len>this->GetRealEndMessageLength()) || 
		(row>=this->num_processors) || (col>=this->num_processors)) return NV::InvRead;
	
	fpos_t save_pos;
	fgetpos(source_file,&save_pos); // save current position
	
	Line l;
	const char *tmp;
	int m_l;
	
	fsetpos(source_file,&data_pos);
	if (mes_len!=this->begin_message_length)
	{
		for ( ; ; )
		{
			if (!readline(source_file,l))
			{
				fsetpos(source_file,&save_pos);
				return NV::InvRead;
			}
			tmp=strchr(l.Give(),'M');
			if ((tmp!=NULL) && !strncmp(tmp,NEW_MESSAGE_LENGTH_STR,NEW_MESSAGE_LENGTH_STR_len))
			{
				tmp+=NEW_MESSAGE_LENGTH_STR_len;
				m_l=0;
				static_cast<void>(sscanf(tmp,"%d",&m_l));
				if (m_l==mes_len) break;
			}
		}
	}
	for (m_l=0; (m_l<row) && readline(source_file,l); ++m_l) {} // skip (row-1) lines
	if ((m_l<row) || !readline(source_file,l)) // read line number 'row'
	{
		fsetpos(source_file,&save_pos);
		return NV::InvRead;
	}
	tmp=l.Give();
	for (m_l=0; m_l<col; ++m_l) // skip (col-1) values
	{
		tmp=strchr(tmp,'\t');
		if (tmp==NULL)
		{
			fsetpos(source_file,&save_pos);
			return NV::InvRead;
		}
		++tmp;
	}
	char *tmp1=const_cast<char*>(strchr(tmp,(flt_pt=='.')? ',' : '.'));
	if (tmp1!=NULL) *tmp1=flt_pt;
	v=atof(tmp);
	fsetpos(source_file,&save_pos);
	return NV::Success;
}

