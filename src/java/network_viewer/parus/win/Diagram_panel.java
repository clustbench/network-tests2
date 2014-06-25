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

package parus.win;

import javax.swing.JPanel;
import javax.swing.JFrame;
import java.util.Map;
import java.util.HashMap;


import java.awt.Graphics;
import java.awt.Dimension;
import java.awt.Color;

public class Diagram_panel extends JPanel
{
	private Integer current_length;
	private Map<Integer,Double> data;
	Integer[] messages_length;
	Double min_value;
	Double max_value;

	



	public void paint(Graphics g)
	{
		super.paint(g);
		
		Dimension dimension;

		int x_offset_for_axis=50;
		int y_offset_for_axis=20;

		dimension=this.getSize(null);

		if
		(
			(dimension.width<x_offset_for_axis)||
			(dimension.height<y_offset_for_axis)
		)
		{
			return;
		}

		int num_messages=messages_length.length;

		double x_resize_coef=
			(double)(dimension.width-x_offset_for_axis)/
			(messages_length[num_messages-1]-messages_length[0]);

		double y_resize_coef=
			(double)(dimension.height-y_offset_for_axis)/
			(max_value-min_value);

		for(int i=0; i < num_messages-1 ; i++ )
		{
			int point1_x=(int)((messages_length[i]-messages_length[0])*x_resize_coef);
			int point1_y=(int)(((Double)data.get(messages_length[i])-min_value)*y_resize_coef);
			
			int point2_x=(int)((messages_length[i+1]-messages_length[0])*x_resize_coef);
			int point2_y=(int)(((Double)data.get(messages_length[i+1])-min_value)*y_resize_coef);
 
			g.drawLine
			(
				point1_x+x_offset_for_axis,
				dimension.height-y_offset_for_axis-point1_y,
				point2_x+x_offset_for_axis,
				dimension.height-y_offset_for_axis-point2_y
			);

		}
		
		
		for(int i=0; i < num_messages ; i++ )
		{
			int point_x=(int)((messages_length[i]-messages_length[0])*x_resize_coef);
			int point_y=(int)(((Double)data.get(messages_length[i])-min_value)*y_resize_coef);
			if(messages_length[i]==current_length)
			{
				//g.fillOval(point_x-2,dimension.height-point_y+2,4,4);
				g.setColor(Color.RED);
			}
			else
			{
				//g.drawOval(point_x-2,dimension.height-point_y+2,4,4);
				g.setColor(Color.GREEN);
			}

			g.drawLine
			(
				point_x-2+x_offset_for_axis,
				dimension.height-y_offset_for_axis-(point_y-2),
				point_x+2+x_offset_for_axis,
				dimension.height-(point_y+2)-y_offset_for_axis
			);

			g.drawLine
			(
				point_x-2+x_offset_for_axis,
				dimension.height-(point_y+2)-y_offset_for_axis,
				point_x+2+x_offset_for_axis,
				dimension.height-(point_y-2)-y_offset_for_axis
			);
		}
		
		g.setColor(Color.BLACK);
		
		g.drawLine
		(
			x_offset_for_axis-1,
			dimension.height-y_offset_for_axis,
			x_offset_for_axis-1,
			0
		);
		
		g.drawLine
		(
			x_offset_for_axis-1,
			dimension.height-y_offset_for_axis,
			dimension.width,
			dimension.height-y_offset_for_axis
		);

		String str;

		g.setColor(Color.BLUE);

		str=""+min_value;
		g.drawString(str,0,dimension.height-y_offset_for_axis);
		
		str=""+max_value;
		g.drawString(str,0,10);

		str=""+messages_length[0];
		g.drawString(str,x_offset_for_axis,dimension.height-5);

		str=""+messages_length[num_messages-1];
		g.drawString(str,dimension.width-(str.length()*8),dimension.height-5);
	}
	
	public Diagram_panel(Map<Integer,Double> d,Integer cur_length)
	{
		super();

		Object arr[] = d.keySet().toArray();
		Object current;
		boolean changes=true;

		this.setBackground(Color.WHITE);
		
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
		
		data=d;
		current_length=cur_length;
		
		min_value=(Double)data.get(messages_length[0]);
		max_value=(Double)data.get(messages_length[0]);
		for(int i=0;i < messages_length.length; i++)
		{
			Double cur=(Double)data.get(messages_length[i]);
			if(min_value > cur) min_value=cur;
			if(max_value < cur) max_value=cur;
		}


	}

	public void set_current_length(Integer cur)
	{
		current_length=cur;
	}

	public static void main(String args[])
	{
		Map<Integer,Double> map=new HashMap<Integer,Double>();

		for(int i=0; i < 100 ; i++ )
		{
			map.put(new Integer(i*i),new Double((double)((i+5)*i)/(i+1)));
		}
		
		Diagram_panel p = new Diagram_panel(map,new Integer(0));
		JFrame frame=new JFrame("Hello world");
		frame.setSize(300,300);
		frame.getContentPane().add(p);
		frame.setVisible(true);		

	}

}

