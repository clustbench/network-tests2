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

//import parus.data.Matrix;

import java.util.ArrayList;
import java.util.Map;
import java.util.HashMap;

import java.io.File;
import java.io.IOException;

public class Network_data
{
	Data_reader reader;

	private int num_processors;
	private int num_messages; // Now it contains number of actually read matrices
	
	/*
	 * Test parameters
	 */
	private String test_type;
	private String data_type;
	private int begin_message_length;
	private int end_message_length;
	private int step_length;
	private int noise_message_length;
	private int num_noise_messages;
	private int num_noise_processes;
	private int num_repeats;
	
	/**
	 * Window size corresponds to the number of matrices in window.
	 */
	private int window_size;
		
	private String   host_names[];
	private Map<Integer,Matrix> test_data;

	private Integer messages_length[];

	public Network_data(File file, int win_size ) throws IOException
	{
		String str;
		
		window_size=win_size;
		
		
		reader=new Data_reader(file);
		
		if(!reader.test_word("processors"))
			throw new IOException("Bad file format: expected 'processors'");
		
		str=reader.read_word();
		num_processors = new Integer(str).intValue();

		boolean correct_flag=false;
		
		/*
		 * Code for old file format
		 */
		str=reader.read_word();
		if(str.equals("num"))
		{
			if(!reader.test_word("messages"))
			throw new IOException("Bad file format: expected 'messages'in 'num messages'");
			
			str=reader.read_word();
			
			begin_message_length=-1;
			end_message_length=-1;
			step_length=-1;
			test_type="\"old_file_format\"";
			noise_message_length=-1;
			num_noise_messages=-1;
			num_noise_processes=-1;
			num_repeats=-1;
			
			host_names=new  String[num_processors];		
			for(int i=0; i< num_processors; i++)
			{
				host_names[i]="(Unknown host)";
			}

			correct_flag=true;
		}
	

		/*
		 * Code for new file format
		 */ 
		if(str.equals("test"))
		{
				
			if(!reader.test_word("type"))
				throw new IOException("Bad file format: expected 'type' in 'test type'");
			
			test_type=reader.read_word();
			
			if(!reader.test_word("data"))
				throw new IOException("Bad file format: expected 'data' in 'data type'");
			else 
			{	
				if(!reader.test_word("type"))
					throw new IOException("Bad file format: expected 'type' in 'data type'");
				data_type=reader.read_word();
			}


			if(!reader.test_word("begin"))
				throw new IOException("Bad file format: expected 'begin' in 'begin message length'");
			if(!reader.test_word("message"))
				throw new IOException("Bad file format: expected 'message' in 'begin message length'");
			if(!reader.test_word("length"))
				throw new IOException("Bad file format: expected 'length' in 'begin message length'");
		
			str=reader.read_word();
			begin_message_length = new Integer(str).intValue();
		

			if(!reader.test_word("end"))
				throw new IOException("Bad file format: expected 'end' in 'end message length' ");
			if(!reader.test_word("message"))
				throw new IOException("Bad file format: expected 'message' in 'end message length' ");
			if(!reader.test_word("length"))
				throw new IOException("Bad file format: expected 'length' in 'end message length'");
	
			str=reader.read_word();
				end_message_length = new Integer(str).intValue();
		

			if(!reader.test_word("step"))
				throw new IOException("Bad file format: expected 'step' in 'step length'");
			if(!reader.test_word("length"))
				throw new IOException("Bad file format: expected 'length' in 'step length'");
		
			str=reader.read_word();
			step_length = new Integer(str).intValue();


			if(!reader.test_word("noise"))
				throw new IOException("Bad file format: expected 'noise' in 'noise message length' ");
			if(!reader.test_word("message"))
				throw new IOException("Bad file format: expected 'message' in 'noise message length'");
			if(!reader.test_word("length"))
				throw new IOException("Bad file format: expected 'length' in 'noise message length'");

			str=reader.read_word();
			noise_message_length = new Integer(str).intValue();


			if(!reader.test_word("number"))
				throw new IOException("Bad file format: expected 'number' in 'number of noise messages'");
			if(!reader.test_word("of"))
				throw new IOException("Bad file format: expected 'of' in 'number of noise messages'");
			if(!reader.test_word("noise"))
				throw new IOException("Bad file format: expected 'noise' in 'number of noise messages'");
			if(!reader.test_word("messages"))
				throw new IOException("Bad file format: expected 'messages' in 'number of noise messages'");
		
			str=reader.read_word();
				num_noise_messages = new Integer(str).intValue();
		

			if(!reader.test_word("number"))
				throw new IOException("Bad file format: expected 'number' in 'number of noise processes' ");
			if(!reader.test_word("of"))
				throw new IOException("Bad file format: expected 'of' in 'number of noise processes' ");
			if(!reader.test_word("noise"))
				throw new IOException("Bad file format: expected 'noise' in 'number of noise processes' ");
			if(!reader.test_word("processes"))
				throw new IOException("Bad file format: expected 'processes' in 'number of noise processes' ");
		
			str=reader.read_word();
			num_noise_processes = new Integer(str).intValue();
		
		
			if(!reader.test_word("number"))
				throw new IOException("Bad file format: expected 'number' in 'number of repeates'");
			if(!reader.test_word("of"))
				throw new IOException("Bad file format: expected 'of' in 'number of repeates'");
			if(!reader.test_word("repeates"))
				throw new IOException("Bad file format: expected 'repeates' in 'number of repeates'");
	
			str=reader.read_word();
			num_repeats = new Integer(str).intValue();


			if(!reader.test_word("hosts:"))
				throw new IOException("Bad file format: expected 'hosts:'");
		
			host_names=new  String[num_processors];
			for(int i=0; i<num_processors; i++)
			{
				host_names[i]=reader.read_word();
			}

			correct_flag=true;

		}
		/*
		 * End code for new file format
		 */

		if(correct_flag==false)
		{
			throw new IOException("Unknown file format: expected 'num messages' for old format or 'test type' for new format of file");
		}

		test_data= new HashMap<Integer,Matrix>();

		num_messages=0;

		System.out.println("Read Matrices");	

		for
		(
			int number_in_window=0;
			number_in_window < window_size;
			number_in_window++
		)
		{

			Integer length;
			Matrix  mtr;

			try
			{
				if(!reader.test_word("Message"))
					throw new IOException("Bad file format: expected 'Message' in 'Message length'");		
				if(!reader.test_word("length"))
					throw new IOException("Bad file format: expected 'length' in 'Message length'");

				str=reader.read_word();
				length=new Integer(str);

				mtr=new Matrix(num_processors,num_processors);
			
				for(int i=0;i<num_processors;i++)
				for(int j=0;j<num_processors;j++)
				{
					str=reader.read_word();
					Double doub=new Double(str);
					mtr.set_element(i,j,doub.doubleValue());
				}
			}
			catch(IOException e)
			{
				Object arr[] = test_data.keySet().toArray();
				Object current;  
				boolean changes=true;

				while(changes)
				{
					changes=false;
					for(int i=1 ; i < arr.length; i++)
					{
						if((Integer)arr[i] < (Integer)arr[i-1])
						{
							current=(Integer)arr[i-1];
							arr[i-1]=arr[i];
							arr[i]=current;
							changes=true;
						}
					}
				}

				messages_length=new Integer[arr.length];
				for(int i=0; i< arr.length;i++)
				{
					messages_length[i]=(Integer)arr[i];
					//System.out.println(messages_length[i].toString());
				}
				
				reader.close();
				return;
			}

			test_data.put(length,mtr);
			System.out.println("Read data for message length: "+length);
			num_messages++;


		}		
		
		Object arr[] = test_data.keySet().toArray();
		Object current;
		boolean changes=true;
		
		while(changes)
		{
			changes=false;
			
			for(int i=1 ; i < arr.length; i++)
			{
				if((Integer)arr[i] < (Integer)arr[i-1])
				{
					current=(Integer)arr[i-1];
					arr[i-1]=arr[i];
					arr[i]=current;
					changes=true;
				}
			}
		}
		
		messages_length=new Integer[arr.length];
		for(int i=0; i< arr.length;i++)
		{
			messages_length[i]=(Integer)arr[i];
			//System.out.println(messages_length[i].toString());
		}

		return;

	}

	public void read_next_window() throws IOException
	{
		
		String str;

		Map<Integer,Matrix> old_test_data=test_data;
		int old_num_messages=num_messages;
		Integer[] old_messages_length=messages_length;

		if(reader.is_open()==false)
		{
			return;
		}

		test_data= new HashMap<Integer,Matrix>();
		num_messages=0;
		
		for
		(
			int number_in_window=0;
			number_in_window < window_size;
			number_in_window++
		)
		{

			Integer length;
			Matrix  mtr;

			try
			{
				if(!reader.test_word("Message"))
					throw new IOException("Bad file format: expected 'Message' in 'Message length'");		
				if(!reader.test_word("length"))
					throw new IOException("Bad file format: expected 'length' in 'Message length'");

				str=reader.read_word();
				length=new Integer(str);

				mtr=new Matrix(num_processors,num_processors);
			
				for(int i=0;i<num_processors;i++)
				for(int j=0;j<num_processors;j++)
				{
					str=reader.read_word();
					Double doub=new Double(str);
					mtr.set_element(i,j,doub.doubleValue());
				}
			}
			catch(IOException e)
			{
				Object arr[] = test_data.keySet().toArray();
				Object current;  
				boolean changes=true;

				while(changes)
				{
					changes=false;
					for(int i=1 ; i < arr.length; i++)
					{
						if((Integer)arr[i] < (Integer)arr[i-1])
						{
							current=(Integer)arr[i-1];
							arr[i-1]=arr[i];
							arr[i]=current;
							changes=true;
						}
					}
				}

				messages_length=new Integer[arr.length];
				for(int i=0; i< arr.length;i++)
				{
					messages_length[i]=(Integer)arr[i];
					//System.out.println(messages_length[i].toString());
				}
				
				reader.close();
				
				if(num_messages==0)
				{
					test_data=old_test_data;
					num_messages=old_num_messages;
					messages_length=old_messages_length;

				}

				return;
			}

			test_data.put(length,mtr);
			System.out.println("Read data for message length: "+length);
			num_messages++;


		}

		Object arr[] = test_data.keySet().toArray();
		Object current;
		boolean changes=true;
		
		while(changes)
		{
			changes=false;
			
			for(int i=1 ; i < arr.length; i++)
			{
				if((Integer)arr[i] < (Integer)arr[i-1])
				{
					current=(Integer)arr[i-1];
					arr[i-1]=arr[i];
					arr[i]=current;
					changes=true;
				}
			}
		}
		
		messages_length=new Integer[arr.length];
		for(int i=0; i< arr.length;i++)
		{
			messages_length[i]=(Integer)arr[i];
			//System.out.println(messages_length[i].toString());
		}
		
		if(num_messages==0)
		{
			test_data=old_test_data;
			num_messages=old_num_messages;
			messages_length=old_messages_length;
		}

		return;

	}

	public String test_parameters_toString()
	{
		String str= new String();
		
		str+="processors "+num_processors+"\n\n";
		str+="test type "+test_type+"\n";
		str+="data_type "+data_type+"\n";
		str+="begin message length "+begin_message_length+"\n";
		str+="end message length "+end_message_length+"\n";
		str+="step length "+step_length+"\n";
		str+="noise message length "+noise_message_length+"\n";
		str+="number of noise messages "+num_noise_messages+"\n";
		str+="number of noise processes "+num_noise_processes+"\n";
		str+="number of repeates "+num_repeats+"\n\n";
		
		return str;

	}

	public String test_parameters_toHTML()
	{
		String str="<html><body><p>\n";
		
		str+="<b>processors</b> "+num_processors+"<br><br>\n";
		str+="<b>test type</b> "+test_type+"<br>\n";
		str+="<b>data type</b> "+data_type+"<br>\n";
		str+="<b>begin message length</b> "+begin_message_length+"<br>\n";
		str+="<b>end message length</b> "+end_message_length+"<br>\n";
		str+="<b>step length</b> "+step_length+"<br>\n";
		str+="<b>noise message length</b> "+noise_message_length+"<br>\n";
		str+="<b>number of noise messages</b> "+num_noise_messages+"<br>\n";
		str+="<b>number of noise processes</b> "+num_noise_processes+"<br>\n";
		str+="<b>number of repeates</b> "+num_repeats+"<br>\n";
		
		return str+"</p></body></html>";

	}public String toString()
	{
		String str=new String();
		
		str+=test_parameters_toString();

		str+="hosts:\n";
		for(int i=0;i < num_processors; i++)
		{
			str+=host_names[i]+'\n';
		}
		str+='\n';

		Object arr[] = test_data.keySet().toArray();
		for(int i=0;i< arr.length; i++)
		{
			Matrix mtr=test_data.get(arr[i]);
			str+="Message length "+arr[i]+"\n";
			for(int j=0;j<num_processors;j++)
			{
				for(int k=0;k<num_processors;k++)
				{
					str+=mtr.get_element(j,k)+" ";
				}
				str+='\n';
			}
			str+='\n';
		}
		
		return str;
	}
	
	public int get_min_read_message_length()
	{	
		if(num_messages==0)
		{
			return -1;
		}
		else
		{
			return messages_length[0].intValue();
		}
	}
	
	public int get_max_read_message_length()
	{
		if(num_messages==0)
		{
			return -1;
		}
		else
		{
			return messages_length[num_messages-1].intValue();
		}

	}

	public double get_max_data_value()
	{
		double max_arr[]=new double[num_messages];
		double max;
		
		for(int i=0; i < num_messages ; i++ )
		{
			Matrix mtr=test_data.get(messages_length[i]);
			max_arr[i]=mtr.get_max_element();
		}

		max=max_arr[0];
		for(int i=0 ; i < num_messages ; i++)
		{
			if(max < max_arr[i]) max=max_arr[i];
		}

		return max;

	}

	public double get_min_data_value()
	{
		double min_arr[]=new double[num_messages];
		double min;
		
		for(int i=0; i < num_messages ; i++ )
		{
			Matrix mtr=test_data.get(messages_length[i]);
			min_arr[i]=mtr.get_min_element();
		}

		min=min_arr[0];
		for(int i=0 ; i < num_messages ; i++)
		{
			if(min > min_arr[i]) min=min_arr[i];
		}

		return min;

	}

	public Matrix get_matrix_by_length(Integer message_length)
	{
		return test_data.get(message_length);
	}

	public Integer[] get_messages_length()
	{
		return messages_length;
	}
	
	public String[] get_host_names()
	{
		return host_names;
	}
	
	public String get_host_name(int proc_number)
	{
		return host_names[proc_number];
	}

	public int get_window_size()
	{
		return window_size;
	}

	public void set_window_size(int win_size)
	{
		window_size=win_size;
		return;
	}

	public Integer get_nearest_message_length(int message_length)
	{
	
		if(message_length <= messages_length[0].intValue())
		{
			return messages_length[0];
		}

		if(message_length >= messages_length[num_messages-1])
		{
			return messages_length[num_messages-1];
		}

		for(int i=0 ; i< num_messages; i++)
		{
			if(message_length <= messages_length[i])
			{
				
				if
				(
				 (messages_length[i]-message_length)/
				 (double)(messages_length[i]-messages_length[i-1]) < 0.5
				)
				{
					return messages_length[i];
				}
				else
				{
					return messages_length[i-1];
				}
			}
		}

		return null;

	}

	public Map<Integer,Double> get_fixed_pair_data(int from,int to)
	{
		Map<Integer,Double> map=new HashMap<Integer,Double>();
		for(int i=0; i < num_messages; i++)
		{
			Matrix m=test_data.get(messages_length[i]);
			map.put(messages_length[i],new Double(m.get_element(from,to)));
		}

		return map;
	}

	public Matrix create_row_matrix(int from)
	{
		Matrix mtr=new Matrix(num_processors,num_messages);

		for(int i=0; i < num_messages ; i++)
		{
			Matrix m=test_data.get(messages_length[i]);
			for(int j=0; j < num_processors ;j++)
			{
				mtr.set_element(j,i,m.get_element(from,j));
			}
		}

		return mtr;
	}

	public Matrix create_column_matrix(int to)
	{
		Matrix mtr=new Matrix(num_processors,num_messages);

		for(int i=0; i < num_messages ; i++)
		{
			Matrix m=test_data.get(messages_length[i]);
			for(int j=0; j < num_processors ;j++)
			{
				mtr.set_element(j,i,m.get_element(j,to));
			}
		}

		return mtr;
	}
}

