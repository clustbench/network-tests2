/*
 *  This file is a part of the PARUS project.
 *  Copyright (C) 2006  Alexey N. Salnikov
 *  
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * 
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this library; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
 * 
 * Alexey N. Salnikov salnikov@cmc.msu.ru
 *
 */

package parus.data;

import java.io.FileReader;
import java.io.File;
import java.io.IOException;

public class Data_reader
{
	public long read_bytes;
	public int string;
	public int column;


	private FileReader reader;
	private boolean open_flag;

	Data_reader(File file) throws IOException
	{
		reader=new FileReader(file);
		read_bytes=0;
		string=0;
		column=0;
		open_flag=true;
		
	}

	public boolean test_word(String word) throws IOException
	{
		int symbol;
		
		do
		{
			symbol=reader.read();
			if(symbol==-1)
			{			
				return false;
			}
			//System.out.write(symbol);
			read_bytes++;
			if(symbol=='\n')
			{
				column=0;
				string++;
			}
		}
		while
		(
			symbol==' '  ||
			symbol=='\t' ||
			symbol=='\r' ||
			symbol=='\n'
		);

		if(symbol!=word.charAt(0)) return false;

		for(int i=1; i < word.length(); i++ )
		{
			symbol=reader.read();
			if(symbol==-1) return false;
			//System.out.write(symbol);
			read_bytes++;
			if(symbol=='\n')
			{
				column=0;
				string++;
			}
			if(symbol!=word.charAt(i)) return false;
		}

		return true;
	}

	public String read_word() throws IOException
	{
		String str=new String();

		int symbol;
		
		do
		{
			symbol=reader.read();
			if(symbol==-1) 
				throw  new IOException("Unexpected End of FILE 1");
			//System.out.write(symbol);

			read_bytes++;
			if(symbol=='\n')
			{
				column=0;
				string++;
			}
		}
		while
		(
			symbol==' '  ||
			symbol=='\t' ||
			symbol=='\r' ||
			symbol=='\n'
		);
		
		while
		(
			symbol!=' '  &&
			symbol!='\t' &&
			symbol!='\r' &&
			symbol!='\n'
		)
		{
			str+=new Character((char)symbol);
			symbol=reader.read();
			if(symbol==-1) 
				throw new IOException("Enexpected End of FILE 2");
			//System.out.write(symbol);
			
			read_bytes++;
			
			if(symbol=='\n')
			{
				column=0;
				string++;
			}
			

		}

		return str;



	}

	public void close() throws IOException
	{
		if(open_flag==true)
		{
			reader.close();
			open_flag=false;
		}

		return;
	}

	public boolean is_open()
	{
		return open_flag;
	}

	public static void main(String args[]) throws IOException
	{
		Data_reader r = new Data_reader(new File("test"));
		System.out.printf("%b\n",r.test_word("Hello"));
		System.out.println(r.read_word());
		r.close();
	}

}

