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

public class Matrix
{
	int size_x;
	int size_y;
	double [] data;
	
	public Matrix(int sx,int sy)
	{
		//System.out.println("new double array sx="+sx+", sy="+sy);
		data = new double [sx*sy];
		size_x=sx;
		size_y=sy;
	}

	public double get_element(int x,int y)
	{
		return data[x*size_y+y];
	}

	public int get_size_x()
	{
		return size_x;
	}
	
	public int get_size_y()
	{
		return size_y;
	}

	public double get_max_element()
	{
		double max=data[0];

		for(int i=0; i < size_x*size_y; i++)
		{
			if(data[i] > max)
			{
				max=data[i];
			}
		}

		return max;
	}

	public double get_min_element()
	{
		double min=data[0];

		for(int i=0; i < size_x*size_y; i++)
		{
			if(data[i] < min)
			{
				min=data[i];
			}
		}

		return min;
	}

	public void set_element(int x,int y,double value)
	{
		data[x*size_y+y]=value;
	}

}

